import collections
import copy

import librosa
import numpy as np

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d
from mido import MidiFile


sr_global = 16000 # Sample rate.
n_fft = 2048 # fft points (samples)
frame_shift = 0.02 # seconds
frame_length = 0.02 # seconds
hop_length = int(sr_global*frame_shift) # samples.
win_length = int(sr_global*frame_length) # samples.
n_mels = 64 # Number of Mel banks to generate
power = 1.2 # Exponent for amplifying the predicted magnitude
n_iter = 100 # Number of inversion iterations
preemphasis = .97 # or None
max_db = 100
ref_db = 20
top_db = 15


def get_spectrograms(y, sr):
    '''Returns normalized log(melspectrogram) and log(magnitude) from `sound_file`.
    Args:
      sound_file: A string. The full path of a sound file.

    Returns:
      mel: A 2d array of shape (T, n_mels) <- Transposed
      mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
 '''
    # Loading sound file
    # y, sr = librosa.load(fpath, sr=sr_global)

    # Trimming
    y, _ = librosa.effects.trim(y, top_db=top_db)

    # Preemphasis
    y = np.append(y[0], y[1:] - preemphasis * y[:-1])

    # stft
    linear = librosa.stft(y=y,
                          n_fft=n_fft,
                          hop_length=hop_length,
                          win_length=win_length)

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T)

    # mel spectrogram
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t)

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize
    mel = np.clip((mel - ref_db + max_db) / max_db, 1e-8, 1)
    mag = np.clip((mag - ref_db + max_db) / max_db, 1e-8, 1)

    # Transpose
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

    return mel, mag

def melspectrogram2wav(mel):
    '''# Generate wave file from spectrogram'''
    # transpose
    mel = mel.T

    # de-noramlize
    mel = (np.clip(mel, 0, 1) * max_db) - max_db + ref_db

    # to amplitude
    mel = np.power(10.0, mel * 0.05)
    m = _mel_to_linear_matrix(sr_global, n_fft, n_mels)
    mag = np.dot(m, mel)

    # wav reconstruction
    wav = griffin_lim(mag)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -preemphasis], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)


def spectrogram2wav(mag):
    '''# Generate wave file from spectrogram'''
    # transpose
    mag = mag.T

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * max_db) - max_db + ref_db

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(mag)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -preemphasis], wav)

    # c
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)



def _mel_to_linear_matrix(sr, n_fft, n_mels):
    m = librosa.filters.mel(sr, n_fft, n_mels)
    m_t = np.transpose(m)
    p = np.matmul(m, m_t)
    d = [1.0 / x if np.abs(x) > 1.0e-8 else x for x in np.sum(p, axis=0)]
    return np.matmul(m_t, np.diag(d))


def griffin_lim(spectrogram):
    '''Applies Griffin-Lim's raw.
    '''
    X_best = copy.deepcopy(spectrogram)
    for i in range(n_iter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, n_fft, hop_length, win_length=win_length)
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y


def invert_spectrogram(spectrogram):
    '''
    spectrogram: [f, t]
    '''
    return librosa.istft(spectrogram, hop_length, win_length=win_length, window="hann")

def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data

def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

def get_normals(vertices, faces):
    '''
    vertices b n 3
    faces f 3
    '''
    verts_normals = torch.zeros_like(vertices)

    vertices_faces = vertices[:, faces]  # b f 3 3

    verts_normals.index_add_(
        1,
        faces[:, 1],
        torch.cross(
            vertices_faces[:, :, 2] - vertices_faces[:, :, 1],
            vertices_faces[:, :, 0] - vertices_faces[:, :, 1],
            dim=2,
        ),
    )
    verts_normals.index_add_(
        1,
        faces[:, 2],
        torch.cross(
            vertices_faces[:, :, 0] - vertices_faces[:, :, 2],
            vertices_faces[:, :, 1] - vertices_faces[:, :, 2],
            dim=2,
        ),
    )
    verts_normals.index_add_(
        1,
        faces[:, 0],
        torch.cross(
            vertices_faces[:, :, 1] - vertices_faces[:, :, 0],
            vertices_faces[:, :, 2] - vertices_faces[:, :, 0],
            dim=2,
        ),
    )

    verts_normals = F.normalize(verts_normals, p=2, dim=2, eps=1e-6)
    # verts_normals = mynormalize(verts_normals, p=2, dim=2, eps=1e-6)

    return verts_normals

def hampel(vals_orig, k=7, t0=3):
    """
    Hampel filter
    :param
        - vals_orig: np.array
        - k: int, size of window. 整数，滤波器窗口的大小（半窗口大小为k/2）
        - t0: float, threshold
    :return
        - vals_filt: np.array
        - outliers_indices: index
    """
    # vals_filt = np.copy(vals_orig)
    outliers_indices = []

    n = len(vals_orig)

    for i in range(k, n - k):
        window = vals_orig[i - k:i + k + 1]

        median = np.median(window)
        mad = np.median(np.abs(window - median))

        if np.abs(vals_orig[i] - median) > t0 * mad:
            # vals_filt[i] = median
            outliers_indices.append(i)

    return outliers_indices

def process_miss_data(array, is_missed):
    is_missed = is_missed.copy()
    X = np.array(range(len(array)))
    ValidDataIndex = X[np.where(is_missed == 0)]

    if ValidDataIndex[-1] < len(array) - 1:
        array[ValidDataIndex[-1] + 1:] = array[ValidDataIndex[-1]]
        is_missed[ValidDataIndex[-1] + 1:] = 0

    if ValidDataIndex[0] >= 1:
        array[:ValidDataIndex[0]] = array[ValidDataIndex[0]]
        is_missed[:ValidDataIndex[0]] = 0

    Y_c = array[ValidDataIndex]
    for i in range(array.shape[1]):
        Y_0 = Y_c[:, i]
        IRFunction = interp1d(ValidDataIndex, Y_0, kind='linear')
        Fill_X = X[np.where(is_missed == 1)]
        Fill_Y = IRFunction(Fill_X)
        array[Fill_X, i] = Fill_Y
    return array

def find_unshowed(is_missed, threshold=15):
    is_missed = is_missed.copy()
    unshowed_idx = []
    count_missed = 0
    tmp_idx = []
    for idx, value in enumerate(is_missed):
        if value == 0:
            if count_missed >= threshold:
                unshowed_idx = unshowed_idx + tmp_idx
            count_missed = 0
            tmp_idx = []
        if value == 1:
            count_missed += 1
            tmp_idx.append(idx)
    if count_missed >= threshold:
        unshowed_idx = unshowed_idx + tmp_idx
    return unshowed_idx

def find_showed(is_missed, threshold=15):
    is_missed = is_missed.copy()
    unshowed_idx = []
    count_missed = 0
    tmp_idx = []
    for idx, value in enumerate(is_missed):
        if value == 1:
            if count_missed <= threshold:
                unshowed_idx = unshowed_idx + tmp_idx
            count_missed = 0
            tmp_idx = []
        if value == 0:
            count_missed += 1
            tmp_idx.append(idx)
    if count_missed <= threshold:
        unshowed_idx = unshowed_idx + tmp_idx
    return unshowed_idx

# https://github.com/bytedance/piano_transcription
def read_midi(midi_path):
    """Parse MIDI file.

    Args:
      midi_path: str

    Returns:
      midi_dict: dict, e.g. {
        'midi_event': [
            'program_change channel=0 program=0 time=0',
            'control_change channel=0 control=64 value=127 time=0',
            'control_change channel=0 control=64 value=63 time=236',
            ...],
        'midi_event_time': [0., 0, 0.98307292, ...]}
    """

    midi_file = MidiFile(midi_path)
    ticks_per_beat = midi_file.ticks_per_beat

    assert len(midi_file.tracks) == 2
    """The first track contains tempo, time signature. The second track 
    contains piano events."""

    microseconds_per_beat = midi_file.tracks[0][0].tempo
    beats_per_second = 1e6 / microseconds_per_beat
    ticks_per_second = ticks_per_beat * beats_per_second

    message_list = []

    ticks = 0
    time_in_second = []

    for message in midi_file.tracks[1]:
        message_list.append(str(message))
        ticks += message.time
        time_in_second.append(ticks / ticks_per_second)

    midi_dict = {
        'midi_event': np.array(message_list),
        'midi_event_time': np.array(time_in_second)}

    return midi_dict

# https://github.com/bytedance/piano_transcription
class TargetProcessor(object):
    def __init__(self, segment_seconds, frames_per_second, begin_note,
                 classes_num):
        """Class for processing MIDI events to target.

        Args:
          segment_seconds: float
          frames_per_second: int
          begin_note: int, A0 MIDI note of a piano
          classes_num: int
        """
        self.segment_seconds = segment_seconds
        self.frames_per_second = frames_per_second
        self.begin_note = begin_note
        self.classes_num = classes_num
        self.max_piano_note = self.classes_num - 1

    def process(self, start_time, midi_events_time, midi_events,
                extend_pedal=True, segment_seconds=None, note_shift=0):
        """Process MIDI events of an audio segment to target for training,
        includes:
        1. Parse MIDI events
        2. Prepare note targets
        3. Prepare pedal targets

        Args:
          start_time: float, start time of a segment
          midi_events_time: list of float, times of MIDI events of a recording,
            e.g. [0, 3.3, 5.1, ...]
          midi_events: list of str, MIDI events of a recording, e.g.
            ['note_on channel=0 note=75 velocity=37 time=14',
             'control_change channel=0 control=64 value=54 time=20',
             ...]
          extend_pedal, bool, True: Notes will be set to ON until pedal is
            released. False: Ignore pedal events.

        Returns:
          target_dict: {
            'onset_roll': (frames_num, classes_num),
            'offset_roll': (frames_num, classes_num),
            'reg_onset_roll': (frames_num, classes_num),
            'reg_offset_roll': (frames_num, classes_num),
            'frame_roll': (frames_num, classes_num),
            'velocity_roll': (frames_num, classes_num),
            'mask_roll':  (frames_num, classes_num),
            'pedal_onset_roll': (frames_num,),
            'pedal_offset_roll': (frames_num,),
            'reg_pedal_onset_roll': (frames_num,),
            'reg_pedal_offset_roll': (frames_num,),
            'pedal_frame_roll': (frames_num,)}

          note_events: list of dict, e.g. [
            {'midi_note': 51, 'onset_time': 696.64, 'offset_time': 697.00, 'velocity': 44},
            {'midi_note': 58, 'onset_time': 697.00, 'offset_time': 697.19, 'velocity': 50}
            ...]

          pedal_events: list of dict, e.g. [
            {'onset_time': 149.37, 'offset_time': 150.35},
            {'onset_time': 150.54, 'offset_time': 152.06},
            ...]
        """
        if segment_seconds is None:
            segment_seconds = self.segment_seconds
        # ------ 1. Parse MIDI events ------
        # Search the begin index of a segment
        for bgn_idx, event_time in enumerate(midi_events_time):
            if event_time > start_time:
                break
        """E.g., start_time: 709.0, bgn_idx: 18003, event_time: 709.0146"""

        # Search the end index of a segment
        for fin_idx, event_time in enumerate(midi_events_time):
            if event_time > start_time + segment_seconds:
                break
        """E.g., start_time: 709.0, bgn_idx: 18196, event_time: 719.0115"""

        note_events = []
        """E.g. [
            {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44}, 
            {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
            ...]"""

        pedal_events = []
        """E.g. [
            {'onset_time': 696.46875, 'offset_time': 696.62604}, 
            {'onset_time': 696.8063, 'offset_time': 698.50836}, 
            ...]"""

        buffer_dict = {}  # Used to store onset of notes to be paired with offsets
        pedal_dict = {}  # Used to store onset of pedal to be paired with offset of pedal

        # Backtrack bgn_idx to earlier indexes: ex_bgn_idx, which is used for
        # searching cross segment pedal and note events. E.g.: bgn_idx: 1149,
        # ex_bgn_idx: 981
        _delta = int((fin_idx - bgn_idx) * 1.)
        ex_bgn_idx = max(bgn_idx - _delta, 0)

        for i in range(ex_bgn_idx, fin_idx):
            # Parse MIDI messiage
            attribute_list = midi_events[i].split(' ')

            # Note
            if attribute_list[0] in ['note_on', 'note_off']:
                """E.g. attribute_list: ['note_on', 'channel=0', 'note=41', 'velocity=0', 'time=10']"""

                midi_note = int(attribute_list[2].split('=')[1])
                velocity = int(attribute_list[3].split('=')[1])

                # Onset
                if attribute_list[0] == 'note_on' and velocity > 0:
                    buffer_dict[midi_note] = {
                        'onset_time': midi_events_time[i],
                        'velocity': velocity}

                # Offset
                else:
                    if midi_note in buffer_dict.keys():
                        note_events.append({
                            'midi_note': midi_note,
                            'onset_time': buffer_dict[midi_note]['onset_time'],
                            'offset_time': midi_events_time[i],
                            'velocity': buffer_dict[midi_note]['velocity']})
                        del buffer_dict[midi_note]

            # Pedal
            elif attribute_list[0] == 'control_change' and attribute_list[2] == 'control=64':
                """control=64 corresponds to pedal MIDI event. E.g. 
                attribute_list: ['control_change', 'channel=0', 'control=64', 'value=45', 'time=43']"""

                ped_value = int(attribute_list[3].split('=')[1])
                if ped_value >= 64:
                    if 'onset_time' not in pedal_dict:
                        pedal_dict['onset_time'] = midi_events_time[i]
                else:
                    if 'onset_time' in pedal_dict:
                        pedal_events.append({
                            'onset_time': pedal_dict['onset_time'],
                            'offset_time': midi_events_time[i]})
                        pedal_dict = {}

        # Add unpaired onsets to events
        for midi_note in buffer_dict.keys():
            note_events.append({
                'midi_note': midi_note,
                'onset_time': buffer_dict[midi_note]['onset_time'],
                'offset_time': start_time + segment_seconds,
                'velocity': buffer_dict[midi_note]['velocity']})

        # Add unpaired pedal onsets to data
        if 'onset_time' in pedal_dict.keys():
            pedal_events.append({
                'onset_time': pedal_dict['onset_time'],
                'offset_time': start_time + segment_seconds})

        # Set notes to ON until pedal is released
        if extend_pedal:
            note_events = self.extend_pedal(note_events, pedal_events)

        # Prepare targets
        frames_num = int(round(segment_seconds * self.frames_per_second)) + 1

        onset_roll = np.zeros((frames_num, self.classes_num), dtype=np.float32)
        offset_roll = np.zeros((frames_num, self.classes_num), dtype=np.float32)
        reg_onset_roll = np.ones((frames_num, self.classes_num), dtype=np.float32)
        reg_offset_roll = np.ones((frames_num, self.classes_num), dtype=np.float32)
        frame_roll = np.zeros((frames_num, self.classes_num), dtype=np.float32)
        velocity_roll = np.zeros((frames_num, self.classes_num), dtype=np.float32)
        mask_roll = np.ones((frames_num, self.classes_num), dtype=np.float32)
        """mask_roll is used for masking out cross segment notes"""

        pedal_onset_roll = np.zeros(frames_num, dtype=np.float32)
        pedal_offset_roll = np.zeros(frames_num, dtype=np.float32)
        reg_pedal_onset_roll = np.ones(frames_num, dtype=np.float32)
        reg_pedal_offset_roll = np.ones(frames_num, dtype=np.float32)
        pedal_frame_roll = np.zeros(frames_num, dtype=np.float32)

        # ------ 2. Get note targets ------
        # Process note events to target
        for note_event in note_events:
            """note_event: e.g., {'midi_note': 60, 'onset_time': 722.0719, 'offset_time': 722.47815, 'velocity': 103}"""

            piano_note = np.clip(note_event['midi_note'] - self.begin_note + note_shift, 0, self.max_piano_note)
            """There are 88 keys on a piano"""

            if 0 <= piano_note <= self.max_piano_note:
                bgn_frame = int(round((note_event['onset_time'] - start_time) * self.frames_per_second))
                fin_frame = int(round((note_event['offset_time'] - start_time) * self.frames_per_second))

                if fin_frame >= 0:
                    frame_roll[max(bgn_frame, 0): fin_frame + 1, piano_note] = 1

                    offset_roll[fin_frame, piano_note] = 1
                    velocity_roll[max(bgn_frame, 0): fin_frame + 1, piano_note] = note_event['velocity']

                    # Vector from the center of a frame to ground truth offset
                    reg_offset_roll[fin_frame, piano_note] = \
                        (note_event['offset_time'] - start_time) - (fin_frame / self.frames_per_second)

                    if bgn_frame >= 0:
                        onset_roll[bgn_frame, piano_note] = 1

                        # Vector from the center of a frame to ground truth onset
                        reg_onset_roll[bgn_frame, piano_note] = \
                            (note_event['onset_time'] - start_time) - (bgn_frame / self.frames_per_second)

                    # Mask out segment notes
                    else:
                        mask_roll[: fin_frame + 1, piano_note] = 0

        for k in range(self.classes_num):
            """Get regression targets"""
            reg_onset_roll[:, k] = self.get_regression(reg_onset_roll[:, k])
            reg_offset_roll[:, k] = self.get_regression(reg_offset_roll[:, k])

        # Process unpaired onsets to target
        for midi_note in buffer_dict.keys():
            piano_note = np.clip(midi_note - self.begin_note + note_shift, 0, self.max_piano_note)
            if 0 <= piano_note <= self.max_piano_note:
                bgn_frame = int(round((buffer_dict[midi_note]['onset_time'] - start_time) * self.frames_per_second))
                mask_roll[bgn_frame:, piano_note] = 0

                # ------ 3. Get pedal targets ------
        # Process pedal events to target
        for pedal_event in pedal_events:
            bgn_frame = int(round((pedal_event['onset_time'] - start_time) * self.frames_per_second))
            fin_frame = int(round((pedal_event['offset_time'] - start_time) * self.frames_per_second))

            if fin_frame >= 0:
                pedal_frame_roll[max(bgn_frame, 0): fin_frame + 1] = 1

                pedal_offset_roll[fin_frame] = 1
                reg_pedal_offset_roll[fin_frame] = \
                    (pedal_event['offset_time'] - start_time) - (fin_frame / self.frames_per_second)

                if bgn_frame >= 0:
                    pedal_onset_roll[bgn_frame] = 1
                    reg_pedal_onset_roll[bgn_frame] = \
                        (pedal_event['onset_time'] - start_time) - (bgn_frame / self.frames_per_second)

        # Get regresssion padal targets
        reg_pedal_onset_roll = self.get_regression(reg_pedal_onset_roll)
        reg_pedal_offset_roll = self.get_regression(reg_pedal_offset_roll)

        target_dict = {
            'onset_roll': onset_roll, 'offset_roll': offset_roll,
            'reg_onset_roll': reg_onset_roll, 'reg_offset_roll': reg_offset_roll,
            'frame_roll': frame_roll, 'velocity_roll': velocity_roll,
            'mask_roll': mask_roll, 'reg_pedal_onset_roll': reg_pedal_onset_roll,
            'pedal_onset_roll': pedal_onset_roll, 'pedal_offset_roll': pedal_offset_roll,
            'reg_pedal_offset_roll': reg_pedal_offset_roll, 'pedal_frame_roll': pedal_frame_roll
        }

        return target_dict, note_events, pedal_events

    def extend_pedal(self, note_events, pedal_events):
        """Update the offset of all notes until pedal is released.

        Args:
          note_events: list of dict, e.g., [
            {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44},
            {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
            ...]
          pedal_events: list of dict, e.g., [
            {'onset_time': 696.46875, 'offset_time': 696.62604},
            {'onset_time': 696.8063, 'offset_time': 698.50836},
            ...]

        Returns:
          ex_note_events: list of dict, e.g., [
            {'midi_note': 51, 'onset_time': 696.63544, 'offset_time': 696.9948, 'velocity': 44},
            {'midi_note': 58, 'onset_time': 696.99585, 'offset_time': 697.18646, 'velocity': 50}
            ...]
        """
        note_events = collections.deque(note_events)
        pedal_events = collections.deque(pedal_events)
        ex_note_events = []

        idx = 0  # Index of note events
        while pedal_events:  # Go through all pedal events
            pedal_event = pedal_events.popleft()
            buffer_dict = {}  # keys: midi notes, value for each key: event index

            while note_events:
                note_event = note_events.popleft()

                # If a note offset is between the onset and offset of a pedal,
                # Then set the note offset to when the pedal is released.
                if pedal_event['onset_time'] < note_event['offset_time'] < pedal_event['offset_time']:

                    midi_note = note_event['midi_note']

                    if midi_note in buffer_dict.keys():
                        """Multiple same note inside a pedal"""
                        _idx = buffer_dict[midi_note]
                        del buffer_dict[midi_note]
                        ex_note_events[_idx]['offset_time'] = note_event['onset_time']

                    # Set note offset to pedal offset
                    note_event['offset_time'] = pedal_event['offset_time']
                    buffer_dict[midi_note] = idx

                ex_note_events.append(note_event)
                idx += 1

                # Break loop and pop next pedal
                if note_event['offset_time'] > pedal_event['offset_time']:
                    break

        while note_events:
            """Append left notes"""
            ex_note_events.append(note_events.popleft())

        return ex_note_events

    def get_regression(self, input):
        """Get regression target. See Fig. 2 of [1] for an example.
        [1] Q. Kong, et al., High-resolution Piano Transcription with Pedals by
        Regressing Onsets and Offsets Times, 2020.

        input:
          input: (frames_num,)

        Returns: (frames_num,), e.g., [0, 0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.9, 0.7, 0.5, 0.3, 0.1, 0, 0, ...]
        """
        step = 1. / self.frames_per_second
        output = np.ones_like(input)

        locts = np.where(input < 0.5)[0]
        if len(locts) > 0:
            for t in range(0, locts[0]):
                output[t] = step * (t - locts[0]) - input[locts[0]]

            for i in range(0, len(locts) - 1):
                for t in range(locts[i], (locts[i] + locts[i + 1]) // 2):
                    output[t] = step * (t - locts[i]) - input[locts[i]]

                for t in range((locts[i] + locts[i + 1]) // 2, locts[i + 1]):
                    output[t] = step * (t - locts[i + 1]) - input[locts[i]]

            for t in range(locts[-1], len(input)):
                output[t] = step * (t - locts[-1]) - input[locts[-1]]

        output = np.clip(np.abs(output), 0., 0.05) * 20
        output = (1. - output)

        return output

def read_frame_roll(midi_events_time, midi_events, start_time, segment_seconds, fps):
    for bgn_idx, event_time in enumerate(midi_events_time):
        if event_time > start_time:
            break
    for fin_idx, event_time in enumerate(midi_events_time):
        if event_time > start_time + segment_seconds:
            break
    note_events = []
    pedal_events = []

    buffer_dict = {}  # Used to store onset of notes to be paired with offsets
    pedal_dict = {}  # Used to store onset of pedal to be paired with offset of pedal

    # Backtrack bgn_idx to earlier indexes: ex_bgn_idx, which is used for
    # searching cross segment pedal and note events. E.g.: bgn_idx: 1149,
    # ex_bgn_idx: 981
    _delta = int((fin_idx - bgn_idx) * 1.)
    ex_bgn_idx = max(bgn_idx - _delta, 0)

    for i in range(ex_bgn_idx, fin_idx):
        # Parse MIDI messiage
        attribute_list = midi_events[i].split(' ')

        # Note
        if attribute_list[0] in ['note_on', 'note_off']:
            """E.g. attribute_list: ['note_on', 'channel=0', 'note=41', 'velocity=0', 'time=10']"""

            midi_note = int(attribute_list[2].split('=')[1])
            velocity = int(attribute_list[3].split('=')[1])

            # Onset
            if attribute_list[0] == 'note_on' and velocity > 0:
                buffer_dict[midi_note] = {
                    'onset_time': midi_events_time[i],
                    'velocity': velocity}

            # Offset
            else:
                if midi_note in buffer_dict.keys():
                    note_events.append({
                        'midi_note': midi_note,
                        'onset_time': buffer_dict[midi_note]['onset_time'],
                        'offset_time': midi_events_time[i],
                        'velocity': buffer_dict[midi_note]['velocity']})
                    del buffer_dict[midi_note]

        # Pedal
        elif attribute_list[0] == 'control_change' and attribute_list[2] == 'control=64':
            """control=64 corresponds to pedal MIDI event. E.g. 
            attribute_list: ['control_change', 'channel=0', 'control=64', 'value=45', 'time=43']"""

            ped_value = int(attribute_list[3].split('=')[1])
            if ped_value >= 64:
                if 'onset_time' not in pedal_dict:
                    pedal_dict['onset_time'] = midi_events_time[i]
            else:
                if 'onset_time' in pedal_dict:
                    pedal_events.append({
                        'onset_time': pedal_dict['onset_time'],
                        'offset_time': midi_events_time[i]})
                    pedal_dict = {}

    # Add unpaired onsets to events
    for midi_note in buffer_dict.keys():
        note_events.append({
            'midi_note': midi_note,
            'onset_time': buffer_dict[midi_note]['onset_time'],
            'offset_time': segment_seconds + start_time,
            'velocity': buffer_dict[midi_note]['velocity']})


    # Prepare targets
    frames_num = int(round(segment_seconds * fps))

    frame_roll = np.zeros((frames_num, 88), dtype=np.float32)


    # ------ 2. Get note targets ------
    # Process note events to target
    for note_event in note_events:
        """note_event: e.g., {'midi_note': 60, 'onset_time': 722.0719, 'offset_time': 722.47815, 'velocity': 103}"""

        piano_note = np.clip(note_event['midi_note'] - 21, 0, 87)
        """There are 88 keys on a piano"""

        if 0 <= piano_note <= 87:
            bgn_frame = int(round((note_event['onset_time'] - start_time) * fps))
            fin_frame = int(round((note_event['offset_time'] - start_time) * fps))

            if fin_frame >= 0:
                frame_roll[max(bgn_frame, 0): fin_frame + 1, piano_note] = 1



    return frame_roll