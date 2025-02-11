# Copyright 2019-2025 The ASReview Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import logging
import multiprocessing as mp
import os
import socket
import time
import webbrowser
from pathlib import Path
from threading import Timer

import requests
import waitress
from rich.console import Console

import asreview as asr
from asreview._deprecated import DeprecateAction
from asreview._deprecated import mark_deprecated_help_strings
from asreview.webapp._task_manager.task_manager import run_task_manager
from asreview.webapp.app import create_app
from asreview.webapp.utils import asreview_path
from asreview.webapp.utils import get_project_path
from asreview.webapp.utils import get_projects

# Host name
HOST_NAME = os.getenv("ASREVIEW_LAB_HOST", "localhost")
PORT_NUMBER = os.getenv("ASREVIEW_LAB_PORT", 5000)


def _check_port_in_use(host, port):
    logging.info(f"Checking if host and port are available :: {host}:{port}")
    host = host.replace("https://", "").replace("http://", "")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def _open_browser(start_url):
    Timer(1, lambda: webbrowser.open_new(start_url)).start()


def _check_for_update():
    """Check if there is an update available."""

    try:
        r = requests.get("https://pypi.org/pypi/asreview/json")
        r.raise_for_status()
        latest_version = r.json()["info"]["version"]
        if latest_version != asr.__version__ and "+" not in asr.__version__:
            return True, latest_version

        return False, latest_version
    except Exception:
        pass


def lab_entry_point(argv):
    """Entry point for the ASReview LAB webapp.

    This function is called when the `asreview lab` command is used.

    Parameters
    ----------
    argv: list
        Command line arguments.

    Examples
    --------
    >>> lab_entry_point(["--port", "5000"])
    Serving ASReview LAB at http://localhost:5000/

    Two examples of how to set the secret key for secure sessions:
    >>> ASREVIEW_LAB_SECRET_KEY="my-secret" asreview lab
    >>> asreview lab --secret-key "my-secret"
    """

    parser = _lab_parser()
    mark_deprecated_help_strings(parser)
    args = parser.parse_args(argv)

    # check for update
    if not args.skip_update_check:
        _check_for_update()

    app = create_app(config_path=args.config_path)

    # override config with command line arguments
    if args.secret_key:
        app.config["SECRET_KEY"] = args.secret_key

    if args.salt:
        app.config["SALT"] = args.salt

    # By default, the application is authenticated but lab is not.
    # Behavior:
    # ENV var for auth | CLI par for auth | App is authenticated
    # ==========================================================
    #       True       |        True      |         True
    #       True       |        False     |         True
    #       False      |        True      |         True
    #       False      |        False     |         False
    #       None       |        True      |         True
    #       None       |        False     |         False
    # ==========================================================
    app.config["AUTHENTICATION"] = (
        app.config.get("AUTHENTICATION", False) or args.authentication
    )

    if app.testing:
        return app

    # NO MORE APP CONFIGURATION BELOW THIS LINE
    # =========================================

    # clean all projects
    # TODO@{Casper}: this needs a little bit
    # of work, we need to access all sub-folders
    if args.clean_all_projects:
        print("Cleaning all project files.")
        for project in get_projects():
            project.clean_tmp_files()
        print("Done")
        return

    # clean project by project_id
    # TODO@{Casper}: cleaning without a user context
    # is meaningless -> I think we should remove this
    # option
    if args.clean_project is not None:
        print(f"Cleaning project file '{args.clean_project}'.")
        asr.Project(get_project_path(args.clean_project)).clean_tmp_files()
        print("Done")
        return

    # if port is already taken find another one
    port = args.port
    original_port = port
    while _check_port_in_use(args.host, port) is True:
        port = int(port) + 1
        if port - original_port >= args.port_retries:
            raise ConnectionError(
                "Could not find an available port \n"
                "to launch ASReview LAB. Last port \n"
                f"was {str(port)}"
            )

    protocol = "https://" if args.certfile and args.keyfile else "http://"
    start_url = f"{protocol}{args.host}:{port}/"

    console = Console()
    console.print("\n\nASReview LAB is starting up [red]<3[/red]\n\n")
    console.print("[bold]Information about your application[/bold]\n")

    host_str = f"[bold]URL:[/bold] {start_url}"
    if original_port != port:
        host_str += f" [yellow][Port {original_port} was already in use][/yellow]"

    version_str = f"[bold]Version:[/bold] {asr.__version__}"
    update_available = False
    if not args.skip_update_check:
        update_available, latest_version = _check_for_update()
        if not update_available:
            version_str += (
                f" [red][Update available: {asr.__version__} -> {latest_version}][/red]"
            )

    console.print(host_str)
    console.print(version_str, highlight=False)
    console.print(f"[bold]Local projects folder:[/bold] {asreview_path()}")

    if update_available:
        console.print(
            "\n\n[bold]Update for ASReview LAB is available![/bold]\n"
            "Run `pip install --upgrade asreview` to update."
        )

    console.print(
        "\n\nMake regular backups of the ASReview projects folder to prevent data loss."
    )
    if not args.no_browser:
        _open_browser(start_url)
        console.print(f"\nIf your browser doesn't open, navigate to {start_url}.\n\n\n")

    console.print("Press [bold]Ctrl+C[/bold] to exit.\n\n")

    # spin up task manager
    start_event = mp.Event()
    process = mp.Process(
        target=run_task_manager,
        args=(
            app.config.get("TASK_MANAGER_WORKERS", None),
            app.config.get("TASK_MANAGER_HOST", None),
            app.config.get("TASK_MANAGER_PORT", None),
            app.config.get("TASK_MANAGER_VERBOSE", False),
            start_event,
        ),
    )
    process.start()

    # wait for the process to spin up
    start_time = time.time()
    while not start_event.is_set():
        time.sleep(0.1)
        if time.time() - start_time > 5:
            console.print(
                "\n\n[red]Error: unable to startup the model server.[/red]\n\n"
            )
            return

    try:
        waitress.serve(app, host=args.host, port=port, threads=6)
    except KeyboardInterrupt:
        console.print("\n\nShutting down server.\n\n")


def _lab_parser():
    # parse arguments if available
    parser = argparse.ArgumentParser(
        prog="lab",
        description="ASReview LAB - Active learning for Systematic Reviews.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--clean-project",
        dest="clean_project",
        type=str,
        help="Safe cleanup of temporary files in project.",
    )

    parser.add_argument(
        "--clean-all-projects",
        dest="clean_all_projects",
        action="store_true",
        help="Safe cleanup of temporary files in all projects.",
    )

    parser.add_argument(
        "--ip",
        default=HOST_NAME,
        type=str,
        action=DeprecateAction,
        help="The IP address the server will listen on. Use the --host argument.",
    )

    parser.add_argument(
        "--host",
        default=HOST_NAME,
        type=str,
        help="The IP address the server will listen on.",
    )

    parser.add_argument(
        "--port",
        default=PORT_NUMBER,
        type=int,
        help="The port the server will listen on.",
    )

    parser.add_argument(
        "--enable-auth",
        dest="authentication",
        default=False,
        action="store_true",
        help="Enable authentication.",
    )

    parser.add_argument(
        "--secret-key",
        type=str,
        help="Secret key for authentication.",
    )

    parser.add_argument(
        "--salt",
        type=str,
        help="When using authentication, a salt code is needed for hasing passwords.",
    )

    parser.add_argument(
        "--config-path",
        "--flask-configfile",
        type=Path,
        help="Path to a TOML file containing ASReview parameters for authentication.",
    )

    parser.add_argument(
        "--no-browser",
        dest="no_browser",
        action="store_true",
        help="Do not open ASReview LAB in a browser after startup.",
    )

    parser.add_argument(
        "--port-retries",
        dest="port_retries",
        default=50,
        type=int,
        help="The number of additional ports to try if the"
        "specified port is not available.",
    )

    parser.add_argument(
        "--certfile",
        default="",
        type=str,
        help="The full path to an SSL/TLS certificate file. "
        "Use dedicated WSGI server (e.g. gUnicorn) instead to make use of TLS. "
        "See https://flask.palletsprojects.com/en/3.0.x/deploying/",
    )

    parser.add_argument(
        "--keyfile",
        default="",
        type=str,
        help="The full path to a private key file for usage with SSL/TLS. "
        "Use dedicated WSGI server (e.g. gUnicorn) instead to make use of TLS. "
        "See https://flask.palletsprojects.com/en/3.0.x/deploying/",
    )

    parser.add_argument(
        "--skip-update-check",
        dest="skip_update_check",
        action="store_true",
        help="Skip checking for updates.",
    )

    return parser
