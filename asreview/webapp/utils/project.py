from copy import deepcopy
import json
import logging
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import numpy as np
from flask import current_app as app

from asreview import __version__ as asreview_version
from asreview.webapp.sqlock import SQLiteLock
from asreview.webapp.utils.paths import *
from asreview.data import ASReviewData
from asreview.review.factory import get_reviewer
from asreview.review.base import get_pool_idx
from asreview.webapp.utils.paths import get_project_file_path, get_iteration_path, get_pool_path
from asreview.webapp.utils.paths import get_result_path
from asreview.webapp.utils.paths import get_labeled_path, get_active_path, asreview_path, get_data_file_path, get_lock_path
from asreview.state.utils import open_state
from asreview.config import LABEL_NA


def get_active_iteration_index(project_id):
    """Get the content of the activity file.

    Arguments
    ---------
    project_id: str
        The id of the current project.
    """

    with open(get_active_path(project_id), "r") as fp:
        result = json.load(fp)

    return int(result.get("current_open"))


def init_project(
        project_id,
        project_name=None,
        project_description=None,
        project_authors=None
):
    """Initialize the necessary files specific to the web app."""

    if not project_id and not isinstance(project_id, str) \
            and len(project_id) >= 3:
        raise ValueError("Project name can't be None or empty string")

    # get the directory with the projects
    project_dir = asreview_path() / project_id

    if project_dir.exists():
        raise ValueError("Project already exists")

    try:
        project_dir.mkdir()

        fp_data = project_dir / "data"
        fp_data.mkdir()

        # create a file with project info
        with open(project_dir / "project.json", "w") as fp:
            json.dump({
                'version': asreview_version,  # todo: Fail without git?
                'id': project_id,
                'name': project_name,
                'description': project_description,
                'authors': project_authors
            }, fp)

        with open(project_dir / "active.json", "w") as fp:
            json.dump({"current_open": 0}, fp)

        asr_kwargs = deepcopy(app.config['asr_kwargs'])  # remove config
        with open(project_dir / "kwargs.json", "w") as fp:
            json.dump(asr_kwargs, fp)

        # data relevant to first iteration
        get_iteration_path(project_id, 0).mkdir()

        # make a copy of the arguments to the state file
        asr_kwargs['log_file'] = str(get_result_path(project_id, 0))

    except Exception as err:
        # remove all generated folders and raise error
        shutil.rmtree(project_dir)
        raise err


def add_dataset_to_project(project_id, file_name):
    """Add file path to the project file.

    Add file to data subfolder and fill the pool of iteration 0.
    """

    project_file_path = get_project_file_path(project_id)

    # open the projects file
    with open(project_file_path, "r") as f_read:
        project_dict = json.load(f_read)

    # add path to dict (overwrite if already exists)
    project_dict["dataset_path"] = file_name

    with open(project_file_path, "w") as f_write:
        json.dump(project_dict, f_write)

    # fill the pool of the first iteration
    pool_indices = read_data(project_id).record_ids
    np.random.shuffle(pool_indices)

    with open(get_pool_path(project_id, 0), "w") as fp:
        json.dump(pool_indices.tolist(), fp)

    # make a empty qeue for the items to label
    with open(get_labeled_path(project_id, 0), "w") as fp:
        json.dump([], fp)


def read_data(project_id):
    """Get ASReviewData object of the dataset"""
    dataset = get_data_file_path(project_id)
    return ASReviewData.from_file(dataset)


def get_paper_data(project_id,
                   paper_id,
                   return_title=True,
                   return_authors=True,
                   return_abstract=True
                   ):
    """Get the title/authors/abstract for a paper."""
    as_data = read_data(project_id)
    paper_id = int(paper_id)
    paper_data = {}
    if return_title and as_data.title is not None:
        paper_data['title'] = as_data.title[paper_id]
    if return_authors and as_data.authors is not None:
        paper_data['authors'] = as_data.authors[paper_id]
    if return_abstract and as_data.abstract is not None:
        paper_data['abstract'] = as_data.abstract[paper_id]
    # paper_data['dataset'] = get_data_file_path(project_id)
    return paper_data


def get_document(paper_id,
                 return_title=True,
                 return_authors=True,
                 return_abstract=True
                 ):
    """Get the title/authors/abstract for a paper."""
    as_data = read_data(project_id)
    paper_id = int(paper_id)
    paper_data = {}
    if return_title and as_data.title is not None:
        paper_data['title'] = as_data.title[paper_id]
    if return_authors and as_data.authors is not None:
        paper_data['authors'] = as_data.authors[paper_id]
    if return_abstract and as_data.abstract is not None:
        paper_data['abstract'] = as_data.abstract[paper_id]
    paper_data['dataset'] = get_data_file_path(project_id)
    return paper_data


def get_documents(paper_ids, *args, **kwargs):
    """Get the title/authors/abstract for a paper."""

    return [get_document(paper_id, *args, **kwargs) for paper_id in paper_ids]


def get_instance(project_id):
    """Get a new instance to review.

    Arguments
    ---------
    project_id: str
        The id of the current project.
    """

    fp_lock = get_project_path(project_id) / "lock.sqlite"

    with SQLiteLock(fp_lock, blocking=True, lock_name="active"):

        # get the path the active pool
        i_current = get_active_iteration_index(project_id)
        pool_path = get_pool_path(project_id, i_current)

        # open the pool and load the indices
        with open(pool_path, "r") as fp:
            pool_idx = json.load(fp)

    logging.info(f"Requesting {pool_idx[0]} from iteration {i_current} and project {project_id}")
    return pool_idx[0]


def get_new_labels(project_id, i):
    """Get all the newly labeled papers from the file.

    Make sure to lock the "active" lock.
    """
    # if active_dir is None:
    #     active_dir = get_active_iteration_path(*args, **kwargs)

    try:
        with open(get_labeled_path(project_id, i), "r") as fp:
            labeled = json.load(fp)
    except FileNotFoundError:
        labeled = []
    return labeled


def get_statistics(project_id):
    fp_lock = get_lock_path(project_id)

    with SQLiteLock(fp_lock, blocking=True, lock_name="active"):
        # get the index of the active iteration
        active_id = get_active_iteration_index(project_id)
        state_fp = get_result_path(project_id, active_id)
        inclusions = []
        try:
            with open_state(state_fp) as state:
                for query_i in range(state.n_queries()):
                    try:
                        incs = state.get("inclusions", query_i).tolist()
                        inclusions.extend(incs)
                    except (KeyError, IndexError):
                        pass
        except FileNotFoundError:
            pass

        # get the next labels to extend
        new_labels = get_new_labels(project_id, active_id)

    # append the new labels to the list of labels
    inclusions.extend([int(x[1]) for x in new_labels])
    n_papers = len(state.get("labels"))
    n_since_last_inclusion = 0
    for i in reversed(range(len(inclusions))):
        if inclusions[i]:
            break
        n_since_last_inclusion += 1

    n_included = int(np.sum(inclusions))
    stats = {
        "n_included": n_included,
        "n_excluded": len(inclusions) - n_included,
        "n_since_last_inclusion": n_since_last_inclusion,
        "n_papers": n_papers,
        "n_pool": n_papers - len(inclusions)
    }
    return stats


def export_to_string(project_id):
    fp_lock = get_lock_path(project_id)
    as_data = read_data(project_id)
    with SQLiteLock(fp_lock, blocking=True, lock_name="active"):

        # get the index of the active iteration
        active_id = get_active_iteration_index(project_id)
        state_fp = get_result_path(project_id, active_id)
        new_labels = get_new_labels(project_id, active_id)
        try:
            with open_state(state_fp) as state:
                labels, _, _, _ = state.review_state()
                proba = state.proba
        except FileNotFoundError:
            state_fp = get_result_path(project_id, active_id-1)
            new_labels += get_new_labels(project_id, active_id-1)
            try:
                with open_state(state_fp) as state:
                    labels, _, _, _ = state.review_state()
                    proba = state.proba
            except FileNotFoundError:
                labels = np.full(len(as_data), LABEL_NA)
        if proba is None:
            proba = np.flip(np.arange(len(as_data)))

    for idx, inclusion in new_labels:
        labels[idx] = inclusion
    pool_idx = np.where(labels == LABEL_NA)[0]
    one_idx = np.where(labels == 1)[0]
    zero_idx = np.where(labels == 0)[0]
    proba_order = np.argsort(-proba[pool_idx, 1])
    ranking = np.concatenate(
        (one_idx, pool_idx[proba_order], zero_idx), axis=None)
    return as_data.to_file(fp=None, labels=labels, ranking=ranking)


def label_instance(project_id, paper_i, label, retrain_model=True):
    """Label a paper after reviewing the abstract.

    """

    fp_lock = get_lock_path(project_id)

    with SQLiteLock(fp_lock, blocking=True, lock_name="active"):

        # get the index of the active iteration
        i_current = get_active_iteration_index(project_id)

        if int(label) in [0, 1]:
            move_label_from_pool_to_labeled(
                project_id, i_current, paper_i, label
            )
        else:
            move_label_from_labeled_to_pool(
                project_id, i_current, paper_i, label
            )

    if retrain_model:
        # Update the model (if it isn't busy).
        run_command = f"python -m asreview web_run_model '{project_id}'"
        subprocess.Popen(shlex.split(run_command))


def move_label_from_pool_to_labeled(project_id, i_current, paper_i, label):

    # load the papers from the pool
    with open(get_pool_path(project_id, i_current), "r") as fp:
        pool_idx = [int(x) for x in json.load(fp)]

    # Remove the paper from the pool.
    try:
        pool_idx.remove(int(paper_i))
        with open(get_pool_path(project_id, i_current), "w") as fp:
            json.dump(pool_idx, fp)
    except (IndexError, ValueError):
        logging.info(f"Failed to remove {paper_i} from the pool.")

    # Add the paper to the reviewed papers.
    labeled = get_new_labels(project_id, i_current)
    label_dict = {l[0]: [l[1], i] for i, l in enumerate(labeled)}
    if paper_i in label_dict:
        labeled[label_dict[paper_i][1]] = [paper_i, label]
    else:
        labeled.append([paper_i, label])

    # write the papers to the label dataset
    with open(get_labeled_path(project_id, i_current), "w") as fp:
        json.dump(labeled, fp)

    print("length pool: ", len(pool_idx))
    print("length labeled: ", len(labeled))
    print("labeled: ", labeled)


def move_label_from_labeled_to_pool(project_id, i_current, paper_i, label):

    # load the papers from the pool
    with open(get_pool_path(project_id, i_current), "r") as fp:
        pool_list = [int(x) for x in json.load(fp)]

    print(get_labeled_path(project_id, i_current))
    # load the papers to the label dataset
    with open(get_labeled_path(project_id, i_current), "r") as fp:
        labeled_list = json.load(fp)

    labeled_list_new = []

    for item_id, item_label in labeled_list:

        item_id = int(item_id)
        item_label = int(item_label)

        if paper_i == item_id:
            pool_list.append(item_id)
        else:
            labeled_list_new.append([item_id, item_label])

    # write the papers to the label dataset
    with open(get_labeled_path(project_id, i_current), "w") as fp:
        json.dump(labeled_list_new, fp)

    # load the papers from the pool
    with open(get_pool_path(project_id, i_current), "w") as fp:
        json.dump(pool_list, fp)

    print("length pool: ", len(pool_list))
    print("length labeled: ", len(labeled_list_new))
    print("labeled: ", labeled_list_new)
