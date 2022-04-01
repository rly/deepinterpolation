import numpy as np
from argparse import ArgumentParser, RawTextHelpFormatter
import json
import spikeinterface as si
from spikeinterface import extractors as se
from spikeinterface import comparison as sc
from spikeinterface import toolkit as st
import h5py

SAMPLING_FREQUENCY = 30000


def load_di_recording_file(file):
    f_input = h5py.File(file, 'r')
    data = f_input['/data'][:].T
    recording = se.NumpyRecording(data, SAMPLING_FREQUENCY)
    return recording


def load_raw_recording_file(file):
    data = np.memmap(file, mode='r', dtype=np.int16, order='C')
    data = np.reshape(data, (-1, 384))
    recording = se.NumpyRecording(data, SAMPLING_FREQUENCY)
    return recording


def get_recording_metrics(x, name):
    metrics = {}
    for i, val in enumerate(x):
        metrics[name + '_' + str(i)] = val
    return metrics


def compare_recordings(raw_recording_file, di_recording_file):
    '''Calculate stddev over specified duration
    '''
    # maximum number of seconds to compute stddev
    max_seconds = 5
    buffer_seconds = 0.1
    raw_recording = load_raw_recording_file(raw_recording_file)
    di_recording = load_raw_recording_file(raw_recording_file)
    rate = raw_recording.get_sampling_frequency()
    num_frames = raw_recording.get_num_frames()

    if num_frames > rate*(buffer_seconds+max_seconds):
        start_frame = int(rate*buffer_seconds)
        end_frame = start_frame + int(rate*max_seconds)
    else:
        start_frame = int(buffer_seconds*rate)
        end_frame = num_frames - start_frame

    raw_data = raw_recording.get_traces(
        start_frame=start_frame, end_frame=end_frame)
    raw_std = np.std(raw_data, axis=0)

    di_data = di_recording.get_traces(
        start_frame=start_frame, end_frame=end_frame)
    di_std = np.std(di_data, axis=0)

    metrics = get_recording_metrics(raw_std, 'raw_std')
    metrics.update(get_recording_metrics(di_std, 'di_std'))

    return metrics, raw_recording, di_recording


def get_sorting_stats(cmp, name):
    ''' Get best matched unit true positives (tp), false positives (fp), 
    and false negatives (fn)
    '''
    perf = cmp.get_confusion_matrix()
    fp = perf.loc['FP'][0]
    fn = perf['FN'][0]
    tp = perf.iloc[1, 1]
    return {name + '_tp': tp, name + '_fp': fp, name + '_fn': fn}


def compare_sortings(raw_sorting_file, di_sorting_file, gt_sorting_file):
    ''' Compare sortings and report accuracy based metrics
    '''
    # load data
    raw_sorting = se.KiloSortSortingExtractor(raw_sorting_file)
    di_sorting = se.KiloSortSortingExtractor(di_sorting_file)
    gt_unit_spk_times = np.load(gt_sorting_file)

    # # make gt sorting object
    gt_sorting = se.NumpySorting.from_times_labels([gt_unit_spk_times],
                                                   [0]*len(gt_unit_spk_times),
                                                   sampling_frequency=30000)

    # compare sortings against ground truth
    cmp_gt_raw = sc.compare_sorter_to_ground_truth(
        gt_sorting, raw_sorting, exhaustive_gt=True)
    raw_stats = get_sorting_stats(cmp_gt_raw, 'raw_')
    cmp_gt_di = sc.compare_sorter_to_ground_truth(
        gt_sorting, di_sorting, exhaustive_gt=True)
    di_stats = get_sorting_stats(cmp_gt_di, 'di')
    metrics = {}
    metrics.update(raw_stats)
    metrics.update(di_stats)
    return metrics, raw_sorting, di_sorting


def calculate_unit_metrics(recording, sorting, metric_names, cache_waveform_folder='waveforms', sorting_name='sorting'):
    '''Calculate unit metrics like SNR or ISI_violations
    '''
    we = si.extract_waveforms(recording, sorting,
                              cache_waveform_folder, load_if_exists=True,
                              ms_before=1, ms_after=2., max_spikes_per_unit=500,
                              n_jobs=1, chunk_size=30000)
    unit_metrics_df = st.compute_quality_metrics(we, metric_names=metric_names)
    unit_metrics_df = unit_metrics_df.add_prefix(sorting_name + '_')
    return unit_metrics_df.to_dict()


def run(raw_recording_file, di_recording_file, raw_sorting_file, di_sorting_file,
        gt_sorting_file, save_file):

    # returns an estimate of the stdev
    recording_metrics, recording_raw, recording_di = compare_recordings(
        raw_recording_file, di_recording_file)
    # returns tp, fp and fn
    sorting_metrics, sorting_raw, sorting_di = compare_sortings(
        raw_sorting_file, di_sorting_file, gt_sorting_file)
    # compute all metrics by passing None
    metric_names = None
    raw_unit_metrics = calculate_unit_metrics(recording_raw, sorting_raw,
                                              metric_names, sorting_name='raw')
    di_unit_metrics = calculate_unit_metrics(recording_di, sorting_di,
                                             metric_names, sorting_name='di')

    # combine metrics
    metrics = {}
    metrics.update(recording_metrics)
    metrics.update(sorting_metrics)
    metrics.update(raw_unit_metrics)
    metrics.update(di_unit_metrics)

    # dump metrics as json file
    with open(save_file, 'w') as file:
        json.dump(metrics, file)


if __name__ == '__main__':
    parser = ArgumentParser(description='Script to compare two recordings and sortings',
                            formatter_class=RawTextHelpFormatter)
    parser.add_argument('--raw_recording_file',
                        type=str, help='Raw recording file')
    parser.add_argument('--di_recording_file', type=str,
                        help='Deepinterpolation recording file')
    parser.add_argument('--raw_sorting_file', type=str,
                        help='Raw sorting file')
    parser.add_argument('--di_sorting_file', type=str,
                        help='Deepinteroplation sorting file')
    parser.add_argument('--gt_sorting_file', type=str,
                        help='Ground truth spikes file')
    parser.add_argument('--save_file', type=str,
                        default='metrics.json', help='Where to save metrics')
    args = parser.parse_args()
    raw_recording_file = args.raw_recording_file
    di_recording_file = args.di_recording_file
    raw_sorting_file = args.raw_sorting_file
    di_sorting_file = args.di_sorting_file
    gt_sorting_file = args.gt_sorting_file
    save_file = args.save_file

    run(raw_recording_file, di_recording_file,
        raw_sorting_file, di_sorting_file,
        gt_sorting_file, save_file)
