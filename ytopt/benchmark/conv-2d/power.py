import argparse
import glob
import os
import subprocess
import time

import pandas as pd
import numpy as np


def delete_ppcg_artifacts():
    for f in glob.glob("*.cu"):
        os.remove(f)
    for f in glob.glob("*.h"):
        os.remove(f)


def get_version():
    if not is_smi_available() and is_jetpack_available():
        return "jetpack"
    res = subprocess.run(
        ["nvidia-smi --query-gpu=gpu_name --format=csv,noheader | head -n 1"], shell=True, stdout=subprocess.PIPE)
    version = res.stdout.decode('utf-8').strip('\n')
    return version


def is_smi_available():
    res1 = subprocess.run(["nvidia-smi"], shell=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("checking if SMI is available")
    if "not found" in res1.stderr.decode('utf-8').strip('\n'):
        print("nvidia-smi not available on target platform")
    return False


def is_jetpack_available():
    res = subprocess.run(["dpkg -l | grep -i \'jetpack\'"],
                         shell=True, stdout=subprocess.PIPE)
    if "Jetpack" in res.stdout.decode('utf-8').strip('\n'):
        return True


def which_compute():
    version = get_version()
    if "A100" in version:
        return "nvidia-smi"
    elif "jetpack" in version:
        return 'tegra'
    else:
        return "mid-tier"


def is_error(target):
    out_res = subprocess.run(
        ["%s" % target], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(out_res.stderr.decode('utf-8'))
    if "CUDA error" in out_res.stderr.decode('utf-8'):
        return 1
    else:
        return 0


def tegra_profile(target, power='off', cuda_visible_devices=str(0)):
    if power == 'off':
       return _tegra_profile_power_off(target, cuda_visible_devices)
    else:
       return _tegra_profile(target, cuda_visible_devices)


def _tegra_profile_power_off(target, cuda_visible_devices=str(0)):
    if is_error(target):
        return False, []
    pid = os.getpid()
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    res = subprocess.run(["sudo", "/usr/local/cuda-10.2/bin/nvprof", "--replay-mode",  "kernel", "--csv",
                          "--system-profiling", "on",  "--normalized-time-unit", "ms", "--log-file", "results/gpu-time-%s.csv" % pid, "./%s" % target, "100"])
    res = subprocess.run(
        ["awk 'NR>4' results/gpu-time-%s.csv > results/gpu-cln-time-%s.csv" % (pid, pid)], shell=True)
    res_time = subprocess.run(
        "awk 'NR>4 {print last} {last=$0}' results/gpu-time-%s.csv | grep -hn 'kernel.' | cut -d ',' -f 5 | paste -s -d + - | bc" % pid, shell=True, stdout=subprocess.PIPE)
    res = subprocess.run(["sudo", "/usr/local/cuda-10.2/bin/nvprof", "--replay-mode",  "kernel", "--csv",
                          "--system-profiling", "on",  "--metrics", "achieved_occupancy,sm_efficiency,warp_execution_efficiency,warps_launched,tex_cache_hit_rate,l2_tex_hit_rate", "--normalized-time-unit", "ms", "--log-file", "results/gpu-stats-%s.csv" % pid, "./%s" % target])
    res = subprocess.run(
        ["awk 'NR>5' results/gpu-stats-{0:}.csv > results/gpu-stats-cln-{0:}.csv".format(pid)], shell=True)

    try:
        time_s = float(res_time.stdout.decode(
            'utf-8').rstrip('\n')) / (10 ** 3)
        df = pd.read_csv('results/gpu-stats-cln-%s.csv' % pid)
        df = df.loc[:, ['Metric Name', 'Avg']]
        df_other = df[df['Metric Name'].isin(
            ['sm_efficiency', 'warp_execution_efficiency', 'tex_cache_hit_rate', 'l2_tex_hit_rate'])]['Avg'].apply(lambda x: float(x.split('%')[0]))
        df_num = df[df['Metric Name'].isin(
            ['achieved_occupancy'])]['Avg'].apply(lambda x: float(x))
        df['Avg'] = pd.concat([df_num, df_other]).sort_index()
        grouped_df = df.groupby('Metric Name').mean()
        grouped_df = grouped_df.reset_index()
        grouped_df.columns = ['Metric Name', 'Stats']
        metric_names = np.array(grouped_df['Metric Name'])
        stats = np.array(grouped_df['Stats'])

        def extract_power(filename):
            with open(filename, "r") as f:
                power_list = []
                for line in f:
                    splitted = line.split(' ')
                    for idx, x in enumerate(splitted):
                        if x == 'GPU':
                            power_list.append(
                                int(splitted[idx + 1].split('/')[0]))
                arr = np.asarray(power_list)
                return np.mean(arr), np.amax(arr), np.amin(arr), np.std(arr), np.quantile(arr, 0.25), np.quantile(arr, 0.50), np.quantile(arr, 0.75), np.quantile(arr, 0.20), np.quantile(arr, 0.40), np.quantile(arr, 0.60), np.quantile(arr, 0.80)
        avg_power = 'None'
        max_power = 'None'
        min_power = 'None'
        quant_25 = 'None'
        quant_50 = 'None'
        quant_75 = 'None'
        quant_20 = 'None'
        quant_40 = 'None'
        quant_60 = 'None'
        quant_80 = 'None'
        metric_names = np.append(np.asarray(
            ['QUANT_25', 'QUANT_50', 'QUANT_75', 'QUANT_20', 'QUANT_40', 'QUANT_60', 'QUANT_80']), metric_names)
        return True, [metric_names, time_s, avg_power, min_power, max_power, quant_25, quant_50, quant_75, quant_20, quant_40, quant_60, quant_80, stats]
    except Exception as e:
        return False, []


def _tegra_profile(target, cuda_visible_devices=str(0)):
    if is_error(target):
        return False, []
    pid = os.getpid()
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    proc = subprocess.run(
        ["sudo /usr/bin/tegrastats --start --interval 10 --logfile power-consumption-%s.csv" % pid], shell=True)
    res = subprocess.run(["sudo", "/usr/local/cuda-10.2/bin/nvprof", "--replay-mode",  "kernel", "--csv",
                          "--system-profiling", "on",  "--normalized-time-unit", "ms", "--log-file", "results/gpu-time-%s.csv" % pid, "./%s" % target, "100"])
    proc = subprocess.run(["sudo /usr/bin/tegrastats --stop"], shell=True)
    res = subprocess.run(
        ["awk 'NR>4' results/gpu-time-%s.csv > results/gpu-cln-time-%s.csv" % (pid, pid)], shell=True)
    res_time = subprocess.run(
        "awk 'NR>4 {print last} {last=$0}' results/gpu-time-%s.csv | grep -hn 'kernel.' | cut -d ',' -f 5 | paste -s -d + - | bc" % pid, shell=True, stdout=subprocess.PIPE)
    res = subprocess.run(["sudo", "/usr/local/cuda-10.2/bin/nvprof", "--replay-mode",  "kernel", "--csv",
                          "--system-profiling", "on",  "--metrics", "achieved_occupancy,sm_efficiency,warp_execution_efficiency,warps_launched,tex_cache_hit_rate,l2_tex_hit_rate", "--normalized-time-unit", "ms", "--log-file", "results/gpu-stats-%s.csv" % pid, "./%s" % target])
    res = subprocess.run(
        ["awk 'NR>5' results/gpu-stats-{0:}.csv > results/gpu-stats-cln-{0:}.csv".format(pid)], shell=True)

    try:
        time_s = float(res_time.stdout.decode(
            'utf-8').rstrip('\n')) / (10 ** 3)
        df = pd.read_csv('results/gpu-stats-cln-%s.csv' % pid)
        df = df.loc[:, ['Metric Name', 'Avg']]
        df_other = df[df['Metric Name'].isin(
            ['sm_efficiency', 'warp_execution_efficiency', 'tex_cache_hit_rate', 'l2_tex_hit_rate'])]['Avg'].apply(lambda x: float(x.split('%')[0]))
        df_num = df[df['Metric Name'].isin(
            ['achieved_occupancy'])]['Avg'].apply(lambda x: float(x))
        df['Avg'] = pd.concat([df_num, df_other]).sort_index()
        grouped_df = df.groupby('Metric Name').mean()
        grouped_df = grouped_df.reset_index()
        grouped_df.columns = ['Metric Name', 'Stats']
        metric_names = np.array(grouped_df['Metric Name'])
        stats = np.array(grouped_df['Stats'])

        def extract_power(filename):
            with open(filename, "r") as f:
                power_list = []
                for line in f:
                    splitted = line.split(' ')
                    for idx, x in enumerate(splitted):
                        if x == 'GPU':
                            power_list.append(
                                int(splitted[idx + 1].split('/')[0]))
                arr = np.asarray(power_list)
                return np.mean(arr), np.amax(arr), np.amin(arr), np.std(arr), np.quantile(arr, 0.25), np.quantile(arr, 0.50), np.quantile(arr, 0.75), np.quantile(arr, 0.20), np.quantile(arr, 0.40), np.quantile(arr, 0.60), np.quantile(arr, 0.80)
        avg_p, max_p, min_p, std_p, quant_25_p, quant_50_p, quant_75_p, quant_20_p, quant_40_p, quant_60_p, quant_80_p = extract_power(
            'power-consumption-%s.csv' % pid)
        avg_power = avg_p
        max_power = max_p
        min_power = min_p
        std_power = std_p
        quant_25 = quant_25_p
        quant_50 = quant_50_p
        quant_75 = quant_75_p
        quant_20 = quant_20_p
        quant_40 = quant_40_p
        quant_60 = quant_60_p
        quant_80 = quant_80_p
        metric_names = np.append(np.asarray(
            ['QUANT_25', 'QUANT_50', 'QUANT_75', 'QUANT_20', 'QUANT_40', 'QUANT_60', 'QUANT_80']), metric_names)
        return True, [metric_names, time_s, avg_power, min_power, max_power, quant_25, quant_50, quant_75, quant_20, quant_40, quant_60, quant_80, stats]
    except Exception as e:
        return False, []


def nvidia_smi_profile(target, power='off', cuda_visible_devices=str(0)):
    if power == 'off':
        return _nvidia_smi_profile_power_off(target, cuda_visible_devices)
    else:
        return _nvidia_smi_profile(target, cuda_visible_devices)


def _nvidia_smi_profile_power_off(target, cuda_visible_devices=str(0)):
    if is_error(target):
        return False, []
    pid = os.getpid()
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    res = subprocess.run(["sudo /shared/centos7/nsight/2021.3.1/ncu --metrics gpu__time_duration.sum,lts__t_requests_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_read.avg.pct_of_peak_sustained_elapsed,lts__t_requests_srcunit_tex_op_write.sum,lts__t_sectors_srcunit_tex_op_write.sum,lts__t_sectors_srcunit_tex_op_write.avg.pct_of_peak_sustained_elapsed,lts__t_sectors.avg.pct_of_peak_sustained_elapsed,l1tex__lsu_writeback_active.avg.pct_of_peak_sustained_active,sm__warps_active.avg.pct_of_peak_sustained_active,sm__warps_launched.avg,smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct --print-summary per-kernel --csv --log-file results/gpu-time-%s.csv ./%s" %
                          (pid, target)], shell=True, stdout=subprocess.PIPE)
    res = subprocess.run(
        ["awk 'NR>2' results/gpu-time-{0:}.csv > results/gpu-cln-time-{0:}.csv".format(pid)], shell=True)
    try:
        df = pd.read_csv('results/gpu-cln-time-%s.csv' % pid)
        time_s = df[df['Metric Name'] ==
                    'gpu__time_duration.sum']['Average'].sum()
        df.drop(df.index[df['Metric Name'] ==
                         'gpu__time_duration.sum'], inplace=True)
        grouped_df = df.groupby("Metric Name").mean()['Average']
        grouped_df = grouped_df.reset_index()
        grouped_df.columns = ['Metric Name', 'Stats']
        metric_names = grouped_df['Metric Name'].to_numpy()
        stats = grouped_df['Stats'].to_numpy()
        time_s = np.true_divide(time_s, 10 ** 9)
        avg_power = 'None'
        max_power = 'None'
        min_power = 'None'
        quant_25 = 'None'
        quant_50 = 'None'
        quant_75 = 'None'
        quant_20 = 'None'
        quant_40 = 'None'
        quant_60 = 'None'
        quant_80 = 'None'
        metric_names = np.append(np.asarray(
            ['QUANT_25', 'QUANT_50', 'QUANT_75', 'QUANT_20', 'QUANT_40', 'QUANT_60', 'QUANT_80']), metric_names)
        return True, [metric_names, time_s, avg_power, min_power, max_power, quant_25, quant_50, quant_75, quant_20, quant_40, quant_60, quant_80, stats]
    except Exception as e:
        return False, []


def _nvidia_smi_profile(target, cuda_visible_devices=str(0)):
    if is_error(target):
        return False, []
    pid = os.getpid()
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    proc = subprocess.Popen(["nvidia-smi", "--query-gpu=index,timestamp,power.draw",
                             "--format=csv", "--loop-ms=10", "--filename=power-consumption-%s.csv" % pid])
    res = subprocess.run(["sudo /shared/centos7/nsight/2021.3.1/ncu --metrics gpu__time_duration.sum,lts__t_requests_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_read.avg.pct_of_peak_sustained_elapsed,lts__t_requests_srcunit_tex_op_write.sum,lts__t_sectors_srcunit_tex_op_write.sum,lts__t_sectors_srcunit_tex_op_write.avg.pct_of_peak_sustained_elapsed,lts__t_sectors.avg.pct_of_peak_sustained_elapsed,l1tex__lsu_writeback_active.avg.pct_of_peak_sustained_active,sm__warps_active.avg.pct_of_peak_sustained_active,sm__warps_launched.avg,smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct --print-summary per-kernel --csv --log-file results/gpu-time-%s.csv ./%s" %
                          (pid, target)], shell=True, stdout=subprocess.PIPE)
    re = proc.terminate()
    res = subprocess.run(
        ["awk 'NR>2' results/gpu-time-{0:}.csv > results/gpu-cln-time-{0:}.csv".format(pid)], shell=True)
    try:
        df = pd.read_csv('results/gpu-cln-time-%s.csv' % pid)
        time_s = df[df['Metric Name'] ==
                    'gpu__time_duration.sum']['Average'].sum()
        df.drop(df.index[df['Metric Name'] ==
                         'gpu__time_duration.sum'], inplace=True)
        grouped_df = df.groupby("Metric Name").mean()['Average']
        grouped_df = grouped_df.reset_index()
        grouped_df.columns = ['Metric Name', 'Stats']
        metric_names = grouped_df['Metric Name'].to_numpy()
        stats = grouped_df['Stats'].to_numpy()
        time_s = np.true_divide(time_s, 10 ** 9)
        time.sleep(1)
        df = pd.read_csv('power-consumption-%s.csv' % pid)
        df = df[df["index"] == int(cuda_visible_devices)]
        pwr = df[" power.draw [W]"]
        pwr = pwr.apply(lambda s: float(s.split(' ')[1]))
        avg_power = np.around(pwr.mean(), 2)
        max_power = pwr.max()
        min_power = pwr.min()
        quant_25 = pwr.quantile(0.25)
        quant_50 = pwr.quantile(0.50)
        quant_75 = pwr.quantile(0.75)
        quant_20 = pwr.quantile(0.20)
        quant_40 = pwr.quantile(0.40)
        quant_60 = pwr.quantile(0.60)
        quant_80 = pwr.quantile(0.80)
        metric_names = np.append(np.asarray(
            ['QUANT_25', 'QUANT_50', 'QUANT_75', 'QUANT_20', 'QUANT_40', 'QUANT_60', 'QUANT_80']), metric_names)
        return True, [metric_names, time_s, avg_power, min_power, max_power, quant_25, quant_50, quant_75, quant_20, quant_40, quant_60, quant_80, stats]
    except Exception as e:
        return False, []


def profile(target, power='off', cuda_visible_devices=str(0)):
    if power == 'off':
        return _profile_power_off(target, cuda_visible_devices)
    else:
        return _profile(target, cuda_visible_devices)


def _profile_power_off(target, cuda_visible_devices=str(0)):
    if is_error(target):
        return False, []
    pid = os.getpid()
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    res = subprocess.run(["CUDA_VISIBLE_DEVICES=%s sudo -E /usr/local/NVIDIA-Nsight-Compute/ncu --metrics gpu__time_duration.sum,lts__t_requests_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_read.avg.pct_of_peak_sustained_elapsed,lts__t_requests_srcunit_tex_op_write.sum,lts__t_sectors_srcunit_tex_op_write.sum,lts__t_sectors_srcunit_tex_op_write.avg.pct_of_peak_sustained_elapsed,lts__t_sectors.avg.pct_of_peak_sustained_elapsed,l1tex__lsu_writeback_active.avg.pct_of_peak_sustained_active,sm__warps_active.avg.pct_of_peak_sustained_active,sm__warps_launched.avg,smsp__cycles_active.avg.pct_of_peak_sustained_elapsed,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct --print-summary per-kernel --csv --log-file results/gpu-time-%s.csv ./%s" %
                          (cuda_visible_devices, pid, target)], shell=True, stdout=subprocess.PIPE)
    res = subprocess.run(
        ["awk 'NR>2' results/gpu-time-{0:}.csv > results/gpu-cln-time-{0:}.csv".format(pid)], shell=True)
    try:
        df = pd.read_csv('results/gpu-cln-time-%s.csv' % pid)
        time_s = df[df['Metric Name'] ==
                    'gpu__time_duration.sum']['Average'].sum()
        df.drop(df.index[df['Metric Name'] ==
                         'gpu__time_duration.sum'], inplace=True)
        grouped_df = df.groupby("Metric Name").mean()['Average']
        grouped_df = grouped_df.reset_index()
        grouped_df.columns = ['Metric Name', 'Stats']
        metric_names = grouped_df['Metric Name'].to_numpy()
        stats = grouped_df['Stats'].to_numpy()
        time_s = np.true_divide(time_s, 10 ** 9)
        time.sleep(1)
        avg_power = 'None'
        max_power = 'None'
        min_power = 'None'
        quant_25 = 'None'
        quant_50 = 'None'
        quant_75 = 'None'
        quant_20 = 'None'
        quant_40 = 'None'
        quant_60 = 'None'
        quant_80 = 'None'
        metric_names = np.append(np.asarray(
            ['QUANT_25', 'QUANT_50', 'QUANT_75', 'QUANT_20', 'QUANT_40', 'QUANT_60', 'QUANT_80']), metric_names)
        return True, [metric_names, time_s, avg_power, min_power, max_power, quant_25, quant_50, quant_75, quant_20, quant_40, quant_60, quant_80, stats]
    except Exception as e:
        return False, []


def _profile(target, cuda_visible_devices=str(0)):
    if is_error(target):
        return False, []
    pid = os.getpid()
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    proc = subprocess.Popen(["nvidia-smi", "--query-gpu=index,timestamp,power.draw",
                             "--format=csv", "--loop-ms=10", "--filename=power-consumption-%s.csv" % pid], stdout=subprocess.DEVNULL)
    res = subprocess.run(["CUDA_VISIBLE_DEVICES=%s ./exe.pl %s" % (cuda_visible_devices, target)], shell=True, stdout=subprocess.PIPE)
    re = proc.terminate()
    try:
        df = pd.read_csv('power-consumption-%s.csv' % pid)
        df = df[df["index"] == int(cuda_visible_devices)]
        pwr = df[" power.draw [W]"]
        pwr = pwr.apply(lambda s: float(s.split(' ')[1]))
        avg_power = np.around(pwr.mean(), 2)
        max_power = pwr.max()
        min_power = pwr.min()
        quant_25 = pwr.quantile(0.25)
        quant_50 = pwr.quantile(0.50)
        quant_75 = pwr.quantile(0.75)
        quant_20 = pwr.quantile(0.20)
        quant_40 = pwr.quantile(0.40)
        quant_60 = pwr.quantile(0.60)
        quant_80 = pwr.quantile(0.80)
        # only print average power
        print(avg_power)
        return True
    except Exception as e:
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Power Profiler')
    parser.add_argument('target')
    args = parser.parse_args()
    target = args.target
    profile(target, power='on')
