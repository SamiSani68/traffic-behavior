#valuate and compare the performance of the ByteTrack and DeepSORT tracking algorithms against a ground truth dataset.
#python tracking/comparison.py
import motmetrics as mm
import pandas as pd
import os
import matplotlib.pyplot as plt

original_class_name_to_group_id = {
    'person': 0, 'bicycle': 0, 'motorcycle': 0,
    'car': 1, 'van': 1, 'truck': 2, 'bus': 2
}

def load_class_mappings(labels_path):
    with open(labels_path, 'r') as f:
        original_classes = [line.strip() for line in f if line.strip()]
    original_class_id_to_name = {i + 1: name for i, name in enumerate(original_classes)}
    original_gt_id_to_grouped_id = {
        i: original_class_name_to_group_id.get(name, 2)
        for i, name in original_class_id_to_name.items()
    }
    return original_gt_id_to_grouped_id

group_id_to_name = {0: 'VRU', 1: 'Fast', 2: 'Slow'}

def prepare_df_for_motmetrics(df, is_gt=False):
    df = df.copy()
    if is_gt:
        df['x1'] = df['x']
        df['y1'] = df['y']
        df['x2'] = df['x'] + df['width']
        df['y2'] = df['y'] + df['height']
        return df[['frameid', 'objectid', 'x1', 'y1', 'x2', 'y2', 'class_id']]
    else:
        df['x1'] = df['x_center'] - df['width'] / 2
        df['y1'] = df['y_center'] - df['height'] / 2
        df['x2'] = df['x_center'] + df['width'] / 2
        df['y2'] = df['y_center'] + df['height'] / 2
        return df[['frame', 'track_id', 'x1', 'y1', 'x2', 'y2', 'class_id']].rename(
            columns={'frame': 'frameid', 'track_id': 'objectid'}
        )

def compute_mot_metrics(gt_df, tracker_df, distth=0.5):
    acc = mm.MOTAccumulator(auto_id=True)
    for frame in sorted(gt_df['frameid'].unique()):
        gt_f = gt_df[gt_df['frameid'] == frame]
        tr_f = tracker_df[tracker_df['frameid'] == frame]
        gt_ids = gt_f['objectid'].tolist()
        tr_ids = tr_f['objectid'].tolist()
        gt_boxes = list(zip(gt_f['x1'], gt_f['y1'], gt_f['x2'], gt_f['y2']))
        tr_boxes = list(zip(tr_f['x1'], tr_f['y1'], tr_f['x2'], tr_f['y2']))
        dist = mm.distances.iou_matrix(gt_boxes, tr_boxes, max_iou=distth)
        acc.update(gt_ids, tr_ids, dist)
    mh = mm.metrics.create()
    return mh.compute(acc, metrics=mm.metrics.motchallenge_metrics, name='acc')

plot_results = []

def analyze_video_tracking(video_name, gt_path, bt_path, ds_path, max_frames=200):
    print(f"\n=== Evaluating {video_name} ===")
    try:
        labels_path = os.path.join("tracking/MOT-analyse/gt", video_name, "labels.txt")
        original_gt_id_to_grouped_id = load_class_mappings(labels_path)

        gt_df = pd.read_csv(gt_path, header=None)
        gt_df.columns = ['frameid', 'objectid', 'x', 'y', 'width', 'height',
                         'confidence', 'class_id_original', 'visibility']
        gt_df['class_id'] = gt_df['class_id_original'].map(original_gt_id_to_grouped_id)
        gt_df = gt_df[gt_df['frameid'] <= max_frames]

        gt_obj_count = len(gt_df['objectid'].unique())

        bt_df = pd.read_csv(bt_path)
        ds_df = pd.read_csv(ds_path)
        bt_df['class_id'] = bt_df['class']
        ds_df['class_id'] = ds_df['class']
        bt_df = bt_df[bt_df['frame'] <= max_frames]
        ds_df = ds_df[ds_df['frame'] <= max_frames]

        gt_prepared = prepare_df_for_motmetrics(gt_df, is_gt=True)
        bt_prepared = prepare_df_for_motmetrics(bt_df)
        ds_prepared = prepare_df_for_motmetrics(ds_df)

        trackers = {'ByteTrack': bt_prepared, 'DeepSORT': ds_prepared}

        print("Overall Metrics:")
        for name, df in trackers.items():
            print(f"\n{name}")
            result = compute_mot_metrics(gt_prepared, df, distth=0.5)
            print(result.round(2))
            plot_results.append({
                'video': video_name,
                'tracker': name,
                'idf1': result.loc['acc', 'idf1'],
                'idp': result.loc['acc', 'idp'],
                'idr': result.loc['acc', 'idr'],
                'mota': result.loc['acc', 'mota'],
                'gt_objects': gt_obj_count
            })

        for group_id in sorted(gt_prepared['class_id'].unique()):
            group_name = group_id_to_name.get(group_id, f"Unknown_{group_id}")
            gt_c = gt_prepared[gt_prepared['class_id'] == group_id]
            bt_c = bt_prepared[bt_prepared['class_id'] == group_id]
            ds_c = ds_prepared[ds_prepared['class_id'] == group_id]

            print(f"Class: {group_name}")
            if not gt_c.empty and (not bt_c.empty or not ds_c.empty):
                for name, df in {'ByteTrack': bt_c, 'DeepSORT': ds_c}.items():
                    print(f"{name}")
                    result = compute_mot_metrics(gt_c, df, distth=0.5)
                    print(result.round(2))
            else:
                print("No valid data for this class.")

    except Exception as e:
        print(f" Error for {video_name}: {e}")

video_configs = {
    "A_70m": {
        "gt": "tracking/MOT-analyse/gt/A_70m/gt.txt",
        "bt": "tracking/MOT-analyse/ByteTrack/A_70m_tracks.csv",
        "ds": "tracking/MOT-analyse/DeepSORT/A_70m_deepsort_tracks.csv"
    },
    "B_50m": {
        "gt": "tracking/MOT-analyse/gt/B_50m/gt.txt",
        "bt": "tracking/MOT-analyse/ByteTrack/B_50m_tracks.csv",
        "ds": "tracking/MOT-analyse/DeepSORT/B_50m_deepsort_tracks.csv"
    },
    "B_80m": {
        "gt": "tracking/MOT-analyse/gt/B_80m/gt.txt",
        "bt": "tracking/MOT-analyse/ByteTrack/B_80m_tracks.csv",
        "ds": "tracking/MOT-analyse/DeepSORT/B_80m_deepsort_tracks.csv"
    },
    "C_145m": {
        "gt": "tracking/MOT-analyse/gt/C_145m/gt.txt",
        "bt": "tracking/MOT-analyse/ByteTrack/C_145m_tracks.csv",
        "ds": "tracking/MOT-analyse/DeepSORT/C_145m_deepsort_tracks.csv"
    }
}

for video, files in video_configs.items():
    analyze_video_tracking(video, files["gt"], files["bt"], files["ds"])

# Plotting
if plot_results:
    df_plot = pd.DataFrame(plot_results)
    metrics = ['idf1', 'idp', 'idr', 'mota']
    for metric in metrics:
        plt.figure(figsize=(10, 5))
        for tracker in df_plot['tracker'].unique():
            subset = df_plot[df_plot['tracker'] == tracker]
            plt.plot(subset['video'], subset[metric], marker='o', label=tracker)
        plt.axhline(y=1.0, linestyle='--', color='gray', label='Perfect Score')
        plt.title(f'{metric.upper()} per Video')
        plt.ylabel(metric.upper())
        plt.xlabel("Video")
        plt.ylim(0, 1.05)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{metric}_comparison.png")
        plt.close()

    plt.figure(figsize=(10, 5))
    gt_counts = df_plot.drop_duplicates(subset=['video'])[['video', 'gt_objects']].set_index('video')
    gt_counts.sort_index().plot(kind='bar', legend=False)
    plt.ylabel("Number of GT Objects")
    plt.title("GT Object Counts per Video")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig("gt_object_counts.png")
    plt.close()
