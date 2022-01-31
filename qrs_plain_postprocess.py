
import os
import sys
import math
import copy
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import logging

from data import datasource_physionet as ds

logger = logging.getLogger(__name__)


def config_logger():
    global logger
    LOG_LEVEL = logging.INFO
    logger.setLevel(LOG_LEVEL)
    # logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(LOG_LEVEL)

    format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(format)
    logger.addHandler(ch)
    # logging.basicConfig(format=format, level=logging.DEBUG, datefmt="%H:%M:%S")
    # logger = logging.getLogger(__name__)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(f"{LOG_PATH}/{LOG_FILE}")
    fh.setFormatter(format)
    fh.setLevel(LOG_LEVEL)
    # add the handlers to the logger
    logger.addHandler(fh)


def __sum_non_zero(
    *args
):
    r"""Sum with non-zero output."""
    total_ = 0.0
    for arg_ in args:
        total_ += float(arg_)
    return total_ if total_ > 0 else 0.0001


def score(
    tp=None, fp=None, fn=None
):
    r"""Calculate scores.

    Arguments:
    - tp: true positive
    - fp: false positive
    - fn: false negative

    Returns:
    - ppv: precision
    - se: sensitivity/recall
    - err: error rate
    - f1: F1 score
    """
    se = 100 * tp / __sum_non_zero(tp, fn)
    ppv = 100 * tp / __sum_non_zero(tp, fp)
    err = 100 * (fp+fn) / __sum_non_zero(tp, fp, fn)
    f1 = (2 * se * ppv) / __sum_non_zero(se, ppv)
    return (ppv, se, err, f1)


class RNode():
    r"""A QRS region."""

    def __init__(
        self, start_loc=0, q_loc=0, confidence=0, support=0
    ):
        r"""Initialise RNode."""
        self.start_loc = start_loc
        self.q_loc = q_loc
        self.confidence = confidence
        self.support = support
        self.l_rr = 0
        self.r_rr = 0

    def __str__(
        self
    ):
        r"""Represent as string."""
        return f'{self.start_loc}:{self.confidence}:{self.support}:{self.q_loc}:{self.l_rr}:{self.r_rr}'

    def __repr__(
        self
    ):
        r"""Represent as string."""
        return self.__str__()


def score_q_loc(
    q_preds, q_labels, start=0, q_offset_sec=0.075*2
):
    r"""Score predicted R locations against gold-standard annotations.

    Returns:
        pred_q_locs: list of locations of R-peaks, with -1 if missed any beat.
        pred_missed: list of missed R-peak locations.
        pred_extra: list of extra detected beat locations.

    """
    N_SAMP_ONE_GOOD = Hz * q_offset_sec
    ones_pred, ones_true = q_preds, q_labels
    pred_q_locs = []
    i_pred = 0
    n_pred_missed = 0
    n_pred_extra = 0
    pred_missed, pred_extra = [], []
    MAX_EXTRA_GUESS = 25
    for i_true, t in enumerate(ones_true):
        if i_pred >= len(ones_pred):
            break
        d = abs(ones_pred[i_pred] - t)
        if d <= N_SAMP_ONE_GOOD:
            # pred_scores[i] = start + ones_pred[i]
            pred_q_locs.append(start + ones_pred[i_pred])
            i_pred += 1
        else:
            found_in_extra = False
            for i_skip in range(1, MAX_EXTRA_GUESS):
                if i_pred+i_skip >= len(ones_pred):
                    break
                d = abs(ones_pred[i_pred+i_skip] - t)
                if d <= N_SAMP_ONE_GOOD:
                    'Record q location'
                    pred_q_locs.append(start + ones_pred[i_pred+i_skip])
                    'Keep missed q location'
                    pred_extra.extend(ones_pred[i_pred:i_pred+i_skip])

                    i_pred += i_skip + 1
                    n_pred_extra += i_skip
                    found_in_extra = not found_in_extra
                    break
            if not found_in_extra:
                r"Missed beat."
                pred_q_locs.append(-1)
                n_pred_missed += 1
                pred_missed.append(t)
    return pred_q_locs, pred_missed, pred_extra


def locate_beats(
    pred_masks, idx_sig_start=0
):
    r"""Locate beats (consecutive 1s) and form nodes."""
    out_nodes = []
    i_start_one = -1
    n_consecutive_one = 0
    # pred_masks = df['pred_mask']
    for i in range(pred_masks.shape[0]):
        if pred_masks[i] == 1:
            r"If 1 counting started, don't increment start index."
            i_start_one = i_start_one if i_start_one > -1 else i
            n_consecutive_one += 1
        else:
            r"Reset 1 counting."
            if n_consecutive_one > 0:
                node = RNode()
                node.start_loc = i_start_one + idx_sig_start
                node.confidence = n_consecutive_one
                node.q_loc = math.ceil(node.start_loc + n_consecutive_one // 2)
                out_nodes.append(node)

                if len(out_nodes) > 1:
                    prev_node = out_nodes[-2]
                    node.l_rr = node.q_loc - prev_node.q_loc
                    prev_node.r_rr = node.l_rr

                i_start_one = -1
                n_consecutive_one = 0
    return out_nodes


# def calculate_rr(
#     node_pair
# ):
#     r"""Calculate right rr distance between two nodes."""
#     node1, node2 = node_pair
#     ret_node = copy.copy(node1)
#     ret_node.r_rr = node2.q_loc - node1.q_loc
#     ret_node.l_rr = 0
#     r"Calculate q_loc as well."
#     ret_node.q_loc = math.ceil(ret_node.start_loc + ret_node.confidence // 2)
#     return ret_node


def calculate_rr(
    nodes
):
    r"""Calculate right rr distance between two nodes."""
    prev = None
    for node in nodes:
        if prev is None:
            prev = node
            continue
        prev.r_rr = node.q_loc - prev.q_loc
        prev.l_rr = 0
        prev = node
    return nodes


def array_match(a, b):
    r"""Match sub-array, b, in a long array, a."""
    for i in range(0, len(a)-len(b)+1):
        if a[i:i+len(b)] == b:
            return i
    return None


def recover_node(
    out_nodes, i_node, df
):
    r"""Recover i_node's next missing node, where pred_score > 0.5.
    Exceptionally large RR distance found between i_node and i_node+1."""
    if i_node+1 >= len(out_nodes):
        return []
    node_1 = out_nodes[i_node]
    node_2 = out_nodes[i_node+1]
    i_scan_start = node_1.start_loc + node_1.confidence + 1
    i_scan_stop = node_2.start_loc - 1
    pred_scores_1s = df['preds_1'].iloc[i_scan_start:i_scan_stop].values
    i_candidate_1s = np.nonzero(pred_scores_1s > 0.5)  # nonzero returns tuple.
    if i_candidate_1s is None or i_candidate_1s[0].shape[0] == 0:
        r"No sample found with pred > 0.5."
        return []
    # logger.debug(f"\t[recover_node] 1s loc:{i_scan_start+i_candidate_1s[0]}")
    r"Create and return candidate nodes."
    pred_stream = np.zeros((i_scan_stop-i_scan_start))
    for i in i_candidate_1s:
        pred_stream[i] = 1
    return locate_beats(pred_stream, idx_sig_start=i_scan_start)


def score_nodes(
    subject, out_nodes, file_suffix="loc.csv", annot=None
):
    r"""Score nodes with ref."""
    r_ref = None
    if annot is not None:
        r_ref = annot['r_locs'].values
    else:
        r_ref = pd.read_csv(f"{PRED_PATH}/{subject}.annot")['r_locs'].values
    r_pred = [x.q_loc for x in out_nodes]
    r_loc_pred, r_loc_missed, r_loc_extra = score_q_loc(r_pred, r_ref)
    r_loc_pred = np.array(r_loc_pred)
    logger.debug(
        f"[{subject}] score r_ref:{r_ref.shape}, r_pred:{r_loc_pred.shape}")

    r"In case of pred stream is shorter than ref, append zeros."
    while r_loc_pred.shape[0] < r_ref.shape[0]:
        r_loc_pred = np.hstack((r_loc_pred, [0]))
    pd.DataFrame({
        'r_ref': r_ref,
        'r_pred': r_loc_pred
    }).to_csv(f"{POSTPROCESS_PATH}/{subject}_loc_pred_{file_suffix}.csv")
    pd.DataFrame({
        'r_missed': r_loc_missed
    }).to_csv(f"{POSTPROCESS_PATH}/{subject}_loc_missed_extra_{file_suffix}.csv", mode='a')
    pd.DataFrame({
        'r_extra': r_loc_extra
    }).to_csv(f"{POSTPROCESS_PATH}/{subject}_loc_missed_extra_{file_suffix}.csv", mode='a')


def recover_intermediate_rr(
    out_nodes, df
):
    r"""Recover intermediate node(s)."""
    idx_last_large_rr = -1
    while True:
        nodes_recov = None
        i_node_large_rr = -1
        for i_node, node in enumerate(out_nodes):
            if node.r_rr > MAX_RR_SAMP and i_node > idx_last_large_rr:
                logger.debug(
                    f"Large RR ({node.r_rr}) detected at node:{i_node}, "
                    f"q_loc:{node.q_loc}")
                i_node_large_rr = i_node
                nodes_recov = recover_node(
                    out_nodes, i_node, df)
                break
        if nodes_recov is None or len(nodes_recov) == 0:
            r"No nodes recovered, break while loop."
            break
        # logger.debug(
        #     f"Current {len(out_nodes)} nodes, recovered "
        #     f"{len(nodes_recov)} nodes at {nodes_recov[0].q_loc}.")
        r"Prevent infinite loop, if unacceptable node found."
        idx_last_large_rr = i_node_large_rr
        r"Nodes recovered, add to main node-list."
        count = 0
        for nd in nodes_recov:
            rr_prev = nd.q_loc - out_nodes[i_node_large_rr+count].q_loc
            rr_next = out_nodes[i_node_large_rr+count+1].q_loc - nd.q_loc
            # logger.debug(
            #     f"\t Node-recov iter, update:{count}, "
            #     f"rr_prev:{rr_prev}={nd.q_loc}-{out_nodes[i_node_large_rr+count].q_loc}, "
            #     f"rr_next:{rr_next}={out_nodes[i_node_large_rr+count+1].q_loc}-{nd.q_loc}")
            if rr_prev > MIN_RR_SAMP and rr_next > MIN_RR_SAMP:
                out_nodes[i_node_large_rr+count].r_rr = rr_prev
                out_nodes.insert(i_node_large_rr+1+count, nd)
                count += 1
                logger.debug(
                    f"\t--iter accepted recv node, loc:{nd.q_loc}, "
                    f"confd:{nd.confidence}")
        r"Calculate r-rr attribute."
        # tmp_out_nodes = list(
        #     map(calculate_rr, zip(out_nodes[:-1], out_nodes[1:])))
        # out_nodes.clear()
        # out_nodes.extend(tmp_out_nodes)
        calculate_rr(out_nodes)
    return out_nodes


def filter_less_confident_min_rr(
    nodes, min_rr=None
):
    r"""Filter a adjacent node with less confidence."""
    out_nodes = []
    prev = None
    skip = False
    for node in nodes:
        if prev is None:
            prev = node
            continue
        if skip:
            skip = not skip
            r"Make sure current node is skipped."
            prev = None
            continue
        if prev.r_rr < min_rr and prev.confidence < node.confidence:
            # prev = node if prev.confidence < node.confidence else prev
            prev = node
            r"Skip current node so that it is not considered as prev anymore."
            skip = True
        else:
            prev = node
        out_nodes.append(prev)

    return out_nodes


# def remove_salt_pepper_noise(
#     pred_mask
# ):
#     r"""Salt-&-pepper noise removal."""
#     patterns = [
#         ([1, 1, 0, 1, 1], [1, 1, 1, 1, 1]),
#         ([0, 0, 1, 0, 0], [0, 0, 0, 0, 0]),
#     ]
#     for i_pat, pat_ in enumerate(patterns):
#         while True:
#             found = False
#             match_loc = array_match(pred_mask, pat_[0])
#             if match_loc:
#                 logger.debug(f"pattern-{i_pat} match at {match_loc}")
#                 r"Replace desired pattern."
#                 pred_mask[match_loc:match_loc+len(pat_[1])] = pat_[1]
#                 found = True
#                 continue
#
#             if not found:
#                 break
#     return pred_mask


def match_and_replace(
    preds, start=0, length=None, pattern=None, replace=None
):
    r"""Match pattern."""
    assert pattern and len(pattern) > 1
    length = length if length is not None else len(preds)
    preds_clone = np.array(preds, copy=True)
    ret_idx = []
    i_current = start
    while i_current < length:
        matched = False
        for i_pattern in range(len(pattern)):
            if (i_current+len(pattern[i_pattern]) < length
                and np.array_equal(
                preds_clone[i_current:i_current+len(pattern[i_pattern])],
                pattern[i_pattern]
            )):
                logger.debug(
                    f"[{i_current}] pattern:{pattern[i_pattern]}")
                preds_clone[i_current:i_current
                            + len(pattern[i_pattern])] = replace[i_pattern]
                ret_idx.append(i_current)
                r"Match with first matched rule and exit."
                matched = True
                break
        if matched:
            i_current += len(pattern[i_pattern])
        else:
            i_current += 1
    return preds_clone, ret_idx


def remove_salt_pepper_noise(
    preds
):
    r"""Postprocess and remove salt-n-pepper noise."""
    pattern = [
        [1, 1, 0, 1, 1],
        [1, 1, 0, 1],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0],
    ]
    replace = []
    for i in range(len(pattern)):
        replace.append(
            [pattern[i][0] for _ in range(len(pattern[i]))]
        )
    new_preds, updated_idx = match_and_replace(
        preds, pattern=pattern, replace=replace)
    logger.debug(
        f"  {len(updated_idx)} patterns updated: {updated_idx}, new_preds:{new_preds.shape}")
    if len(updated_idx) == 0:
        return new_preds
    return remove_salt_pepper_noise(new_preds)


def process_pred_mask(
    subject, df, annot=None, persist_verbose=False
):
    r"""Postprocess prediction."""
    # subject = subject[:subject.find('.csv')]

    r"""Postprocess rule: (S): Minimal step.
        Remove Salt-n-pepper noise.
    """
    list_pred_mask = df['pred_mask'].values
    revised_pred_mask = remove_salt_pepper_noise(list_pred_mask.tolist())
    revised_pred_mask = np.array(revised_pred_mask)
    nodes = locate_beats(revised_pred_mask)
    logger.debug(f"[{subject}] {len(nodes)} QRS nodes detected.")
    if persist_verbose:
        pd.DataFrame({
            'start': [node.start_loc for node in nodes],
            'confidence': [node.confidence for node in nodes],
            'r_loc': [node.q_loc for node in nodes],
            'l_rr': [node.l_rr for node in nodes],
            'r_rr': [node.r_rr for node in nodes],
        }).to_csv(f"{POSTPROCESS_PATH}/{subject}_nodes_saltpepper.csv")
    record_score(
        subject, nodes, pp_step='s', annot=annot)

    # nodes = locate_beats(df['pred_mask'])
    nodes = locate_beats(revised_pred_mask)
    logger.debug(f"[{subject}] {len(nodes)} QRS nodes detected.")
    if persist_verbose:
        pd.DataFrame({
            'start': [node.start_loc for node in nodes],
            'confidence': [node.confidence for node in nodes],
            'r_loc': [node.q_loc for node in nodes],
            'l_rr': [node.l_rr for node in nodes],
            'r_rr': [node.r_rr for node in nodes],
        }).to_csv(f"{POSTPROCESS_PATH}/{subject}_nodes.csv")

    r"""Postprocess rule: (A): Moderate step.
        Remove nodes with confidence less than 64ms (6.4 ~6 samples)
    """
    # min_node_confidence = round(MIN_CONFIDENCE_SAMP)
    out_nodes = list(
        filter(lambda x: x.confidence >= MIN_CONFIDENCE_SAMP and x.r_rr > MIN_RR_SAMP, nodes))
    logger.debug(
        f"[{subject}] {len(out_nodes)} confident nodes from {len(nodes)}.")

    r"Calculate r-rr attribute."
    # out_nodes = list(map(calculate_rr, zip(out_nodes[:-1], out_nodes[1:])))
    calculate_rr(out_nodes)
    if persist_verbose:
        pd.DataFrame({
            'start': [node.start_loc for node in out_nodes],
            'confidence': [node.confidence for node in out_nodes],
            'r_loc': [node.q_loc for node in out_nodes],
            'l_rr': [node.l_rr for node in out_nodes],
            'r_rr': [node.r_rr for node in out_nodes],
        }).to_csv(f"{POSTPROCESS_PATH}/{subject}_nodes_a.csv")
    record_score(
        subject, out_nodes, pp_step='a', annot=annot)

    r"""Postprocess rule: (B): Advanced step.
        Remove less confident node, incase two adjacent candidates are less than
        200ms (20 samples) apart.
    """
    # out_nodes = list(
    #     filter(lambda x: x.r_rr >= MIN_RR_SAMP, out_nodes))
    out_nodes = filter_less_confident_min_rr(
        out_nodes, min_rr=MIN_RR_SAMP)
    logger.debug(
        f"[{subject}] {len(out_nodes)} RR-complient nodes.")
    r"Calculate r-rr attribute."
    # out_nodes = list(map(calculate_rr, zip(out_nodes[:-1], out_nodes[1:])))
    calculate_rr(out_nodes)
    if persist_verbose:
        pd.DataFrame({
            'start': [node.start_loc for node in out_nodes],
            'confidence': [node.confidence for node in out_nodes],
            'r_loc': [node.q_loc for node in out_nodes],
            'l_rr': [node.l_rr for node in out_nodes],
            'r_rr': [node.r_rr for node in out_nodes],
        }).to_csv(f"{POSTPROCESS_PATH}/{subject}_nodes_b.csv")
        score_nodes(
            subject, out_nodes, file_suffix="b", annot=annot)
    r"Score predictions"
    record_score(
        subject, out_nodes, pp_step='b', annot=annot)

    r"""Postprocess rule: (C): Further advanced step (not included in the paper)
        Recover a missing beat if RR > 1200ms (120 samples).
    """
    recover_intermediate_rr(
        out_nodes, df)
    logger.debug(
        f"[{subject}] {len(out_nodes)} nodes after node-recovery.")
    r"Calculate r-rr attribute."
    # out_nodes = list(map(calculate_rr, zip(out_nodes[:-1], out_nodes[1:])))
    calculate_rr(out_nodes)
    if persist_verbose:
        pd.DataFrame({
            'start': [node.start_loc for node in out_nodes],
            'confidence': [node.confidence for node in out_nodes],
            'r_loc': [node.q_loc for node in out_nodes],
            'l_rr': [node.l_rr for node in out_nodes],
            'r_rr': [node.r_rr for node in out_nodes],
        }).to_csv(f"{POSTPROCESS_PATH}/{subject}_nodes_c.csv")
    r"Persist final q-locations."
    score_nodes(
        subject, out_nodes, file_suffix="c", annot=annot)
    r"Record score for this subject."
    record_score(
        subject, out_nodes, pp_step='c', annot=annot)


def record_score(
    subject, nodes, pp_step=None, annot=None
):
    r"""Record subject scores for final result."""
    global postprocess_scores
    pp_step = pp_step.upper()
    r_ref = None
    if annot is not None:
        r_ref = annot['r_locs'].values
    else:
        r_ref = pd.read_csv(f"{PRED_PATH}/{subject}.annot")['r_locs'].values
    r_pred = [x.q_loc for x in nodes]
    r_loc_pred, r_loc_missed, r_loc_extra = score_q_loc(
        r_pred, r_ref)

    n_pred_locs, n_pred_missed, n_pred_extra = len(r_loc_pred), \
        len(r_loc_missed), len(r_loc_extra)

    postprocess_scores[f"pp{pp_step}_n_pred_locs"] += n_pred_locs
    postprocess_scores[f"pp{pp_step}_n_pred_missed"] += n_pred_missed
    postprocess_scores[f"pp{pp_step}_n_pred_extra"] += n_pred_extra

    ppv, se, err, f1 = score(
        tp=n_pred_locs-n_pred_missed, fp=n_pred_extra, fn=n_pred_missed)

    logger.info(
        f"@[{subject}] pp_step:{pp_step}, #beats:{n_pred_locs}, "
        f"missed:{n_pred_missed}, wrong:{n_pred_extra}, Se:{se:.02f}, "
        f"PPV:{ppv:.02f}, F1:{f1:.02f}, err:{err:.02f}")


r"Global score dict."
postprocess_scores = {}


def process_record(
    subject, df, reset_scores=False, annot=None, persist_verbose=False
):
    r"""Postprocess specified dataframe."""
    global postprocess_scores
    if reset_scores:
        postprocess_scores.clear()
        PP_NAMES = "SABC"
        logger.info(
            f"MIN_CONFIDENCE_SAMP:{MIN_CONFIDENCE_SAMP}, "
            f"MIN_RR_SAMP:{MIN_RR_SAMP}, "
            f"MAX_RR_SAMP:{MAX_RR_SAMP}")
        for i in range(len(PP_NAMES)):
            postprocess_scores[f"pp{PP_NAMES[i]}_n_pred_locs"] = 0
            postprocess_scores[f"pp{PP_NAMES[i]}_n_pred_missed"] = 0
            postprocess_scores[f"pp{PP_NAMES[i]}_n_pred_extra"] = 0

    logger.debug(f"[{subject}] pred_mask:{df['pred_mask'].shape}")
    process_pred_mask(
        subject, df, annot=annot, persist_verbose=persist_verbose)


def show_process_summary():
    r"Database level metric."
    PP_NAMES = "SABC"
    for i in range(len(PP_NAMES)):
        n_pred_locs, n_pred_missed, n_pred_extra = \
            postprocess_scores[f"pp{PP_NAMES[i]}_n_pred_locs"],\
            postprocess_scores[f"pp{PP_NAMES[i]}_n_pred_missed"],\
            postprocess_scores[f"pp{PP_NAMES[i]}_n_pred_extra"]
        ppv, se, err, f1 = score(
            tp=n_pred_locs-n_pred_missed, fp=n_pred_extra, fn=n_pred_missed)
        logger.info(
            f"@@Summary|{ds.DB_NAMES[IDX_VAL_DB]}, pp_step:{PP_NAMES[i]}, "
            f"#beats:{n_pred_locs}, "
            f"missed:{n_pred_missed}, wrong:{n_pred_extra}, Se:{se:.02f}, "
            f"PPV:{ppv:.02f}, F1:{f1:.02f}, err:{err:.02f}")


def run():
    r"""Simulation entry point."""
    global postprocess_scores
    postprocess_scores.clear()
    PP_NAMES = "SABC"
    logger.info(
        f"MIN_CONFIDENCE_SAMP:{MIN_CONFIDENCE_SAMP}, "
        f"MIN_RR_SAMP:{MIN_RR_SAMP}, "
        f"MAX_RR_SAMP:{MAX_RR_SAMP}")
    for i in range(len(PP_NAMES)):
        postprocess_scores[f"pp{PP_NAMES[i]}_n_pred_locs"] = 0
        postprocess_scores[f"pp{PP_NAMES[i]}_n_pred_missed"] = 0
        postprocess_scores[f"pp{PP_NAMES[i]}_n_pred_extra"] = 0

    for f in os.listdir(PRED_PATH):
        if not f.endswith(".csv"):
            continue
        df = pd.read_csv(f"{PRED_PATH}/{f}")
        pred_mask = df['pred_mask']
        logger.debug(f"[{f}] pred_mask:{pred_mask.shape}")

        subject = f[:f.find('.csv')]
        process_pred_mask(subject, df)

        # break

    r"Database level metric."
    # PP_NAMES = "SABC"
    for i in range(len(PP_NAMES)):
        n_pred_locs, n_pred_missed, n_pred_extra = \
            postprocess_scores[f"pp{PP_NAMES[i]}_n_pred_locs"],\
            postprocess_scores[f"pp{PP_NAMES[i]}_n_pred_missed"],\
            postprocess_scores[f"pp{PP_NAMES[i]}_n_pred_extra"]
        ppv, se, err, f1 = score(
            tp=n_pred_locs-n_pred_missed, fp=n_pred_extra, fn=n_pred_missed)
        logger.info(
            f"@@Summary|{ds.DB_NAMES[IDX_VAL_DB]}, pp_step:{PP_NAMES[i]}, "
            f"#beats:{n_pred_locs}, "
            f"missed:{n_pred_missed}, wrong:{n_pred_extra}, Se:{se:.02f}, "
            f"PPV:{ppv:.02f}, F1:{f1:.02f}, err:{err:.02f}")


r"""Globals"""
Hz = 100
r"64ms fails for INCART (I12,)"
MIN_CONFIDENCE_SAMP = round(64 * Hz / 1000)    # 64ms
MIN_RR_SAMP = round(200 * Hz / 1000)   # 200ms
r"1200ms INCART"
MAX_RR_SAMP = round(1200 * Hz / 1000)   # 1200ms


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Simulation: Seizure detection.')
    parser.add_argument(
        "--i_valdb", required=True, default=1, help="Validation database.")

    START_TIME_TAG = f"{datetime.now():%Y%m%d%H%M%S}"
    args = parser.parse_args()
    IDX_VAL_DB = int(args.i_valdb)
    HOME = '/home/XXX'
    BASE_PATH = (
        f"{HOME}/py_code/runs/qrs_plain_train.py_100Hz_Tcpsc19_ClassicConv_segsz300_scutTrue_lccfg24_lckr11_lcst1_lccg1_blk1_cpblk2_kr5x1_och24x1_cg1x1_20211019182111")
    MODEL_PATH = f'{BASE_PATH}/models'

    VAL_PATH = None
    r"Find val folder."
    for f in os.listdir(f"{BASE_PATH}/validate"):
        if f.find(ds.DB_NAMES[IDX_VAL_DB]) > -1:
            VAL_PATH = f"{BASE_PATH}/validate/{f}"
    if not VAL_PATH:
        print(f"Val path not found for database:{ds.DB_NAMES[IDX_VAL_DB]}")
        exit(0)
    POSTPROCESS_PATH = (
        f"{VAL_PATH}/postprocess/{START_TIME_TAG}_mnC{MIN_CONFIDENCE_SAMP}_"
        f"mnRR{MIN_RR_SAMP}_mxRR{MAX_RR_SAMP}")
    if not os.path.exists(POSTPROCESS_PATH):
        os.makedirs(POSTPROCESS_PATH)

    LOG_PATH = POSTPROCESS_PATH
    PRED_PATH = f"{VAL_PATH}/preds"

    LOG_FILE = (
        f"{sys.argv[0]}_mnC{MIN_CONFIDENCE_SAMP}_mnRR{MIN_RR_SAMP}_"
        f"mxRR{MAX_RR_SAMP}.log")

    config_logger()

    run()
