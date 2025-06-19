import json
import random
import time
import logging

from locust import HttpUser, task, between
from threading import Lock

# Setup logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("locust")

# locust -f ./asreview/webapp/tests/server/locustfile.py --host https://asreview.app

class WebsiteUser(HttpUser):
    wait_time = between(1, 3)

    # Shared class-level counter and lock
    user_counter = 0
    user_counter_lock = Lock()


    def on_start(self):
        # Safely assign a unique index per user
        with WebsiteUser.user_counter_lock:
            self.user_index = WebsiteUser.user_counter
            WebsiteUser.user_counter += 1

        # Deterministic, reusable credentials
        self.email = f"locust_user_{self.user_index}@asreview.app"
        self.password = "Test123!"
        self.name = f"Locust User {self.user_index}"
        self.affiliation = "Load Testing Lab"

        self.client.headers.update({
            "Content-Type": "application/x-www-form-urlencoded"
        })

        self._authenticate()
        self.finished = False
        self.project_id = self._create_project()
        if not self.project_id:
            raise RuntimeError(f"[{self.email}] Project creation failed.")


    def _authenticate(self):
        login_data = {
            "email": self.email,
            "password": self.password
        }

        with self.client.post("/auth/signin", data=login_data, catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 404 and "does not exist" in response.text:
                # User doesn't exist → sign up
                signup_data = {
                    "email": self.email,
                    "password": self.password,
                    "name": self.name,
                    "affiliation": self.affiliation,
                    "public": "1"
                }
                with self.client.post("/auth/signup", data=signup_data, catch_response=True) as signup_response:
                    if signup_response.status_code == 201:
                        signup_response.success()
                    else:
                        signup_response.failure("Signup failed")
            else:
                response.failure("Login failed unexpectedly")


    def _create_project(self):
        with self.client.post(
            "/api/projects/create",
            data={
                "mode": "oracle",
                "benchmark": "synergy:Appenzeller-Herzog_2019"
            },
            catch_response=True
        ) as response:
            if response.status_code == 201:
                response.success()
                project_id = json.loads(response.text)["id"]

                # Now update the review status from "setup" to "review"
                with self.client.put(
                    f"/api/projects/{project_id}/reviews/0",
                    name="/api/projects/:project_id/reviews/0",
                    data={"status": "review"},
                    catch_response=True
                ) as status_response:
                    if status_response.status_code in (200, 201):
                        status_response.success()
                        return project_id
                    else:
                        status_response.failure(
                            f"Failed to update status to 'review': {status_response.status_code}"
                        )
                        return False
            else:
                response.failure(f"Project creation failed: {response.status_code}")
                return False


    @task
    def get_user(self):
        with self.client.get("/auth/user", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code {response.status_code}")


    @task
    def get_invitations(self):
        with self.client.get("/api/invitations", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code {response.status_code}")


    @task
    def get_users(self):
        if not hasattr(self, "project_id") or self.finished:
            return

        with self.client.get(
            f"/api/projects/{self.project_id}/users",
            name="/api/projects/:project_id/users",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Status code {response.status_code}")


    @task
    def stopping_condition(self):
        if not hasattr(self, "project_id") or self.finished:
            return

        with self.client.get(
            f"/api/projects/{self.project_id}/stopping",
            name="/api/projects/:project_id/stopping",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                self.finished = data.get("stop", False)
                value = data.get("value", 0)
                stopper = data.get("id", "unknown")

                logger.info(f"[{self.email}] Stopper: {stopper}, Value: {value}, Stop: {self.finished}")
                response.success()
            else:
                response.failure(f"Failed to check stopping condition: {response.status_code}")


    @task
    def screen_record(self):
        if not hasattr(self, "project_id") or self.finished:
            return

        record_id = None

        with self.client.get(
            f"/api/projects/{self.project_id}/get_record",
            name="/api/projects/:project_id/get_record",
            catch_response=True
        ) as response:
            if response.status_code != 200:
                response.failure(f"Unexpected status code: {response.status_code}")
                return
            
            data = response.json()
            result = data.get("result", None)
            status = data.get("status", "unknown")

            logger.info(f"RESPONSE {result} {status}")

            if result is None:
                if status == "setup":
                    logger.info(f"[{self.email}] No record yet, project in setup mode. Retrying later...")
                    time.sleep(5)  # wait before next retry
                    response.success()
                    return
                else:
                    self.finished = True
                    logger.info(f"[{self.email}] No more records available. Marked as finished.")
                    response.success()
                    return
            else:
                record_id = result.get("record_id")
                response.success()
                logger.info(f"[{self.email}] Got record {record_id} (status: {status})")

        if record_id is not None:
            time.sleep(random.uniform(10, 20))  # Simulate user reading

            label = random.choice([0, 1])
            tags = []
            retrain_model = True

            with self.client.post(
                f"/api/projects/{self.project_id}/record/{record_id}",
                data={
                    "record_id": record_id,
                    "label": label,
                    "tags": json.dumps(tags),
                    "retrain_model": int(retrain_model)
                },
                name="/api/projects/:project_id/record/:record -> label",  # <- This groups the call
                catch_response=True
            ) as label_response:
                if label_response.status_code != 200:
                    label_response.failure(f"Failed to label record {record_id}")
                else:
                    label_response.success()
                    logger.info(f"[{self.email}] Labeled record {record_id} as {label}")
