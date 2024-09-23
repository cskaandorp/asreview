# Copyright 2019-2022 The ASReview Authors. All Rights Reserved.
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

from pathlib import Path

import pandas as pd

from asreview.config import LABEL_NA
from asreview.extensions import load_extension
from asreview.settings import ReviewSettings
from asreview.simulation.simulate import Simulate
from asreview.state.contextmanager import open_state

from asreview.webapp.app import huey

print("\n\nLOADING TASKS MODULE\n\n")


@huey.task(name="run_model")
def run_model(project):
    print("HELLO!!!!!!!")
    import time

    time.sleep(10)

    with open_state(project) as s:
        if not s.exist_new_labeled_records:
            return

        if s.get_results_table("label")["label"].value_counts().shape[0] < 2:
            return

    try:
        settings = ReviewSettings().from_file(
            Path(
                project.project_path,
                "reviews",
                project.reviews[0]["id"],
                "settings_metadata.json",
            )
        )

        as_data = project.read_data()

        feature_model = load_extension(
            "models.feature_extraction", settings.feature_extraction
        )()
        try:
            fm = project.get_feature_matrix(feature_model)
        except FileNotFoundError:
            fm = feature_model.fit_transform(
                as_data.texts, as_data.headings, as_data.bodies, as_data.keywords
            )
            project.add_feature_matrix(fm, feature_model)

        with open_state(project) as state:
            labeled = state.get_results_table(columns=["record_id", "label"])

        y_input = (
            pd.DataFrame({"record_id": as_data.record_ids})
            .merge(labeled, how="left", on="record_id")["label"]
            .fillna(LABEL_NA)
        )

        if settings.balance_strategy is not None:
            balance_model = load_extension(
                "models.balance", settings.balance_strategy
            )()
            balance_model_name = balance_model.name
            X_train, y_train = balance_model.sample(
                fm, y_input, labeled["record_id"].values
            )
        else:
            X_train, y_train = fm, y_input
            balance_model_name = None

        classifier = load_extension("models.classifiers", settings.classifier)()
        classifier.fit(X_train, y_train)
        relevance_scores = classifier.predict_proba(fm)

        query_strategy = load_extension("models.query", settings.query_strategy)()
        ranked_record_ids = query_strategy.query(
            feature_matrix=fm, relevance_scores=relevance_scores
        )

        with open_state(project) as state:
            state.add_last_ranking(
                ranked_record_ids,
                classifier.name,
                query_strategy.name,
                balance_model_name,
                feature_model.name,
                len(labeled),
            )

        project.remove_review_error()

    except Exception as err:
        project.set_review_error(err)
        raise err


@huey.task(name="run_simulation")
def run_simulation(project):
    as_data = project.read_data()

    settings = ReviewSettings().from_file(
        Path(
            project.project_path,
            "reviews",
            project.reviews[0]["id"],
            "settings_metadata.json",
        )
    )

    with open_state(project) as state:
        priors = state.get_priors()["record_id"].tolist()

    feature_model = load_extension(
        "models.feature_extraction", settings.feature_extraction
    )()
    fm = feature_model.fit_transform(
        as_data.texts, as_data.headings, as_data.bodies, as_data.keywords
    )
    project.add_feature_matrix(fm, feature_model)

    if settings.balance_strategy is not None:
        balance_model = load_extension("models.balance", settings.balance_strategy)()
    else:
        balance_model = None

    sim = Simulate(
        fm,
        labels=as_data.labels,
        classifier=load_extension("models.classifiers", settings.classifier)(),
        query_strategy=load_extension("models.query", settings.query_strategy)(),
        balance_strategy=balance_model,
        feature_extraction=feature_model,
    )
    try:
        sim.label(priors, prior=True)
        sim.review()
    except Exception as err:
        project.set_review_error(err)
        raise err

    project.update_review(state=sim, status="finished")
