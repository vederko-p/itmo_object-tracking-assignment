
from collections import defaultdict

import yaml
import numpy as np

from track_3 import track_data


CB_WIDTH = 120
CB_HEIGHT = 100


def read_tracks_file(filepath: str) -> dict:
    with open(filepath, 'r') as file:
        tracks = yaml.safe_load(file)
    return tracks


def collect_gt_tracks(initial_tracks) -> dict:
    all_gt_tracks = defaultdict(list)
    for frame in initial_tracks:
        for gt_data in frame['data']:
            all_gt_tracks[gt_data['cb_id']].append(
                {
                    'frame_id': frame['frame_id'],
                    'bb': [
                        gt_data['x'] - (CB_WIDTH // 2),
                        gt_data['y'] - CB_HEIGHT,
                        gt_data['x'] + CB_WIDTH - (CB_WIDTH // 2),
                        gt_data['y'] + CB_HEIGHT - CB_HEIGHT,
                    ]
                 }
            )
    return all_gt_tracks


def collect_st_tracks(predicted_tracks) -> dict:
    all_st_tracks = defaultdict(list)
    for frame in predicted_tracks:
        for st_data in frame['data']:
            all_st_tracks[st_data['track_id']].append(
                {'frame_id': frame['frame_id'], 'bb': st_data['bounding_box']}
            )
    return all_st_tracks


def calc_iou(bb1: list, bb2: list) -> float:
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou


def calculate_iou_matrix(gt_tracks: dict, st_tracks: dict) -> np.array:
    """Return matrix of (n,m) size, where n=|gt_tracks| and m=|st_tracks|."""
    s = len(gt_tracks), len(st_tracks)
    iou_matrix = np.zeros(s)  # mean spatial interception
    time_matrix = np.zeros(s)  # mean temporal interception
    for gt_index, (gt_id, gt_metadata) in enumerate(gt_tracks.items()):
        for st_index, (st_id, st_metadata) in enumerate(st_tracks.items()):
            # collect frames:
            gt_frames = [int(m['frame_id']) for m in gt_metadata]
            st_frames = [int(m['frame_id']) for m in st_metadata]
            inter_frames = set(gt_frames) & set(st_frames)
            # calculate IoU:
            iou = 0
            for frame in inter_frames:
                iou += calc_iou(
                    gt_metadata[frame-1]['bb'],
                    st_metadata[st_frames.index(frame)]['bb']
                )
            iou_matrix[gt_index, st_index] = (
                    iou / len(inter_frames) if inter_frames else 0
            )
            # calculate time:
            time_matrix[gt_index, st_index] = (
                    len(inter_frames) / len(gt_frames)
            )
    return iou_matrix.round(3), time_matrix.round(3)


def main_metrics(
        iou_matrix: np.array, time_matrix: np.array,
        iou_threshold: float = 0.5, time_threshold: float = 0.5
) -> tuple:
    true_positive = (
            (iou_matrix > iou_threshold) & (time_matrix > time_threshold)
    ).any(axis=1).astype(int).mean()
    return (
        round(iou_matrix.mean(), 3),
        round(time_matrix.mean(), 3),
        round(true_positive, 3)
    )


if __name__ == '__main__':

    gt_tracks = collect_gt_tracks(track_data)
    st_tracks = collect_st_tracks(
        read_tracks_file('st_tracks/tracks_strong.yaml')
    )
    '''for k, v in st_tracks.items():
        print(f'{k}: {len(v)}')'''

    iou_m, time_m = calculate_iou_matrix(gt_tracks, st_tracks)
    # print(iou_m)
    # print(time_m)

    metrics = main_metrics(iou_m, time_m)
    print(f'mean IoU: {metrics[0]}')
    print(f'mean time inter: {metrics[1]}')
    print(f'mean TP: {metrics[2]}')
