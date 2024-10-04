import json
import socket
import time
import threading
from collections import deque

from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from asreview.webapp import asreview_path
from asreview.webapp.queue.models import Base
from asreview.webapp.queue.models import ProjectQueueModel
from asreview.webapp.queue.task_wrapper import RunModelProcess
from asreview.webapp.tasks import run_task


class TaskManager:
    def __init__(self, max_workers=1, host="localhost", port=5555):
        self.pending = set()
        self.max_workers = max_workers

        # set up parameters for socket endpoint
        self.host = host
        self.port = port
        self.message_buffer = deque()
        self.receive_bytes = 1024 # bytes read when receiving messages
        self.timeout = 0.1 # wait for 0.1 seconds for incoming messages

        # set up database
        database_url = f"sqlite:///{asreview_path()}/queue.sqlite"
        engine = create_engine(database_url)
        Base.metadata.create_all(engine)

        Session = sessionmaker(bind=engine)
        self.session = Session()

    @property
    def waiting(self):
        records = self.session.query(ProjectQueueModel).all()
        return [r.project_id for r in records]

    def insert_in_waiting(self, project_id, simulation):
        # remember that there is a unique constraint on project_id
        try:
            new_record = ProjectQueueModel(
                project_id=str(project_id), simulation=bool(simulation)
            )
            self.session.add(new_record)
            self.session.commit()
        except IntegrityError:
            self.session.rollback()

    def is_waiting(self, project_id):
        record = (
            self.session.query(ProjectQueueModel)
            .filter_by(project_id=project_id)
            .first()
        )
        if record is None:
            return False
        else:
            return record

    def is_pending(self, project_id):
        return project_id in self.pending

    def remove_pending(self, project_id):
        if project_id in self.pending:
            self.pending.remove(project_id)

    def move_from_waiting_to_pending(self, project_id):
        record = self.is_waiting(project_id)
        if record:
            try:
                # add to pending
                self.add_pending(project_id)
                # delete
                self.session.delete(record)
                self.session.commit()
            except Exception:
                self.session.rollback()
                # remove from pending
                self.remove_pending(project_id)

    def add_pending(self, project_id):
        if project_id not in self.pending:
            self.pending.add(project_id)

    def __execute_job(self, project_id, simulation):
        try:
            # run the simulation / train task
            p = RunModelProcess(
                func=run_task,
                args=(project_id, simulation),
                host=self.host,
                port=self.port,
            )
            p.start()
            return True
        except Exception as _:
            return False

    def pop_task_queue(self):
        """Moves tasks from the database and executes them in subprocess."""
        # how many slots do I have?
        available_slots = self.max_workers - len(self.pending)

        if available_slots > 0:
            # select first n records
            records = (
                self.session.query(ProjectQueueModel)
                .order_by(ProjectQueueModel.id)
                .limit(available_slots)
                .all()
            )
            # loop over records
            for record in records:
                project_id = record.project_id
                simulation = record.simulation
                # execute job
                if self.__execute_job(project_id, simulation):
                    # move out of waiting and put into pending
                    self.move_from_waiting_to_pending(project_id)

    def _process_buffer(self):
        """Injects messages in the database."""
        while self.message_buffer:
            message = self.message_buffer.popleft()
            action = message.get("action", False)
            project_id = message.get("project_id", False)
            simulation = message.get("simulation", False)

            if action == "insert" and project_id:
                # This will insert into the waiting database if
                # the project isn't there, it will fail gracefully
                # if the project is already waiting
                self.insert_in_waiting(project_id, simulation)

            elif action in ["remove", "failure"] and project_id:
                self.remove_pending(project_id)

    def _handle_incoming_messages(self, conn):
        """Handles incoming traffic."""
        client_buffer = ""
        while True:
            try:
                data = conn.recv(self.receive_bytes)
                if not data:
                    # if client_buffer is full convert to json and
                    # put in buffer
                    if client_buffer != "":
                        # we may be dealing with multiple messages,
                        # update buffer to produce a correct json string
                        client_buffer = "[" + client_buffer.replace("}{", "},{") + "]"
                        # add to buffer
                        self.message_buffer.extend(deque(json.loads(client_buffer)))
                    # client disconnected
                    break

                else:
                    message = data.decode("utf-8")
                    client_buffer += message

            except Exception:
                break

        conn.close()

    def start_manager(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.host, self.port))
        server_socket.listen()

        # Set a timeout
        server_socket.settimeout(0.1)

        while True:
            try:
                # Accept incoming connections with a timeout
                conn, addr = server_socket.accept()
                # Start a new thread to handle the client connection
                client_thread = threading.Thread(
                    target=self._handle_incoming_messages, args=(conn,)
                )
                client_thread.start()

            except socket.timeout:
                # No incoming connections => perform handling queue
                messages = self._process_buffer()
                # Pop tasks from database into 'pending'
                self.pop_task_queue()


def run_task_manager(max_workers, host, port):
    manager = TaskManager(max_workers=max_workers, host=host, port=port)
    manager.start_manager()


if __name__ == "__main__":
    manager = TaskManager(max_workers=2)
    manager.start_manager()
