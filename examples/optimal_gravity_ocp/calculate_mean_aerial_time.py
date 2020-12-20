import ezc3d
import numpy as np
from load_data_filename import load_data_filename

essai = {'DoCi': ['822', '44_1', '44_2', '44_3'],
         'JeCh': ['833_1', '833_2', '833_3', '833_4', '833_5'],
         'BeLa': ['44_1', '44_2', '44_3'],
         'GuSe': ['44_2', '44_3', '44_4'],
         'SaMi': ['821_822_2', '821_822_3',
                  '821_contact_1', '821_contact_2', '821_contact_3',
                  '822_contact_1',
                  '821_seul_1', '821_seul_2', '821_seul_3', '821_seul_4', '821_seul_5']}

times = []
times_44 = []
times_821 = []
times_822 = []
times_833 = []
for subject, trials in essai.items():
    # print('Subject: ', subject)
    data_path = '/home/andre/Optimisation/data/' + subject + '/'
    c3d_path = data_path + 'Essai/'
    for trial in trials:
        data_filename = load_data_filename(subject, trial)
        c3d_name = data_filename['c3d']
        frames = data_filename['frames']

        c3d = ezc3d.c3d(c3d_path + c3d_name)

        frequency = c3d['header']['points']['frame_rate']
        duration = len(frames) / frequency

        times.append(duration)
        if '44' in trial:
            times_44.append(duration)
        if '821' in trial:
            times_821.append(duration)
        if '822' in trial and '821' not in trial:
            times_822.append(duration)
        if '833' in trial:
            times_833.append(duration)

        # print('Trial: ', trial)
        # print('Duration: ', duration)

average_time = np.mean(times)
rms_time = np.sqrt(np.mean([time**2 for time in times]))
std_time = np.std(times)

average_time_44 = np.mean(times_44)
rms_time_44 = np.sqrt(np.mean([time**2 for time in times_44]))
std_time_44 = np.std(times_44)

average_time_821 = np.mean(times_821)
rms_time_821 = np.sqrt(np.mean([time**2 for time in times_821]))
std_time_821 = np.std(times_821)

average_time_822 = np.mean(times_822)
rms_time_822 = np.sqrt(np.mean([time**2 for time in times_822]))
std_time_822 = np.std(times_822)

average_time_833 = np.mean(times_833)
rms_time_833 = np.sqrt(np.mean([time**2 for time in times_833]))
std_time_833 = np.std(times_833)

print('Average time', average_time, rms_time, std_time)
print('Average time 44', average_time_44, rms_time_44, std_time_44)
print('Average time 821', average_time_821, rms_time_821, std_time_821)
print('Average time 822', average_time_822, rms_time_822, std_time_822)
print('Average time 833', average_time_833, rms_time_833, std_time_833)
