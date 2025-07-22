import pickle
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.offline as offline


def aggr(read_path, save_path):
    """聚合指定路径下的所有结果文件"""
    print(f"\nProcessing files in {read_path}")

    # 获取所有文件
    files = [f for f in listdir(read_path) if isfile(join(read_path, f))]
    path_files = [join(read_path, f) for f in files]

    if not files:
        print(f"警告: {read_path} 目录下没有找到文件!")
        return None

    data = []
    for f, path in zip(files, path_files):
        try:
            # 从文件名解析实验参数
            parts = f.split('_')
            if len(parts) < 3:
                print(f"跳过格式不正确的文件: {f}")
                continue

            exp_range = parts[1]
            exp_alg = parts[2].split('.')[0]

            print(f"处理文件: {f} (范围: {exp_range}, 算法: {exp_alg})")

            # 加载数据
            result = pickle.load(open(path, "rb"))

            # 为每条记录添加实验参数
            for r in result:
                r.append(exp_range)
                r.append(exp_alg)
                data.append(r)

        except Exception as e:
            print(f"加载文件 {path} 时出错: {e}")
            continue

    if not data:
        print(f"警告: 没有成功加载任何数据!")
        return None

    # 创建DataFrame，修复重复列名
    df = pd.DataFrame(data=data, columns=["mean_absolute_error1",
                                          "mean_absolute_error2",
                                          "r2_score",
                                          "RRMSE",
                                          "RMSE",
                                          "range",
                                          "alg",
                                          '1',
                                          '2'])

    # 打印数据摘要
    print(f"聚合后的数据包含 {len(df)} 条记录")
    print(f"数据中的算法: {df['alg'].unique()}")
    print(f"数据中的范围: {df['range'].unique()}")

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"已将聚合数据保存到 {save_path}")
        return df
    else:
        return df


def aggr_all(read_path, save_path, data_path, str_class):
    files = [f for f in listdir(read_path) if isfile(join(read_path, f))]
    path_files = [join(read_path,f) for f in files]

    tot_cv = 10*5
    dt = pd.read_csv(data_path)
    X = dt.drop([str_class], axis=1).values
    y = dt[str_class].values

    perceltil_inf = [5*i for i in range(1,9)]
    perceltil_sup = [(100-perceltil_inf[i]) for i in range(len(perceltil_inf))]
    range_high_TG_per = [np.percentile(y, perceltil_sup[i]) for i in range(len(perceltil_sup))]
    range_low_TG_per = [np.percentile(y, perceltil_inf[i]) for i in range(len(perceltil_inf))]
    range_low_TG = np.arange(start=400, stop=650+1, step=25)
    range_high_TG = np.arange(start=900, stop=1150+1, step=25)

    data=[]
    for f, path in zip(files, path_files):
        exp_range = f.split('_')[1]
        exp_alg = f.split('_')[2].split('.')[0]
        if exp_alg in ("high", "low"):
            alg = f.split('_')[3].split('.')[0]
            try:
                result = pickle.load(open(path, "rb" ))
                if(exp_alg == "high"):
                    if("percentil" in read_path.split("/")[3].split("_")):
                        ran = range_high_TG_per * tot_cv
                    else:
                        ran = range_high_TG.tolist() * tot_cv
                else:
                    if("percentil" in read_path.split("/")[3].split("_")):
                        ran = range_low_TG_per * tot_cv
                    else:
                        ran = range_low_TG.tolist() * tot_cv
                result = [j for i in result for j in i]
                [(r.append(ran_i),r.append(alg), data.append(r)) for r, ran_i in zip(result, ran)]
            except Exception as e:
                print(f"Error loading file {path}: {e}")
        else:
            try:
                result = pickle.load(open(path, "rb" ))
                # 检查文件名是否包含"all"或类似标识
                if "all" in exp_range.lower() or "full" in exp_range.lower():
                    [(r.append("all"), r.append(exp_alg), data.append(r)) for r in result]
                else:
                    [(r.append(exp_range), r.append(exp_alg), data.append(r)) for r in result]
            except Exception as e:
                print(f"Error loading file {path}: {e}")

    # 修复重复列名
    df = pd.DataFrame(data=data, columns=["mean_absolute_error1",
                                          "mean_absolute_error2",
                                          "r2_score",
                                          "RRMSE",
                                          "RMSE",
                                          "range",
                                          "alg",
                                          '1',
                                          '2'])
    if save_path != None:
        df.to_csv(save_path, index=False)
        print(f"Saved aggregated data to {save_path}")
        return df
    else:
        return df


def plot_bar(alg, save_path, measure, percentil=False):
    """为单个算法创建箱线图，比较全量数据和特定范围数据"""
    print(f"\n生成 {alg} 的箱线图 (percentil={percentil})")

    title = None
    if percentil:
        title = f"{alg}: All X Range (percentile)"
        save_path = f"{save_path}{alg}_{measure}_allXrange_percentile.html"
        files = {
            'all': '../../result/aggr/result_all_percentil.csv',
            'high': '../../result/aggr/result_high_percentil.csv',
            'low': '../../result/aggr/result_low_percentil.csv'
        }
    else:
        title = f"{alg}: All X Range"
        save_path = f"{save_path}{alg}_{measure}_allXrange.html"
        files = {
            'all': '../../result/aggr/result_all.csv',
            'high': '../../result/aggr/result_high.csv',
            'low': '../../result/aggr/result_low.csv'
        }

    # 打印绝对路径
    print(f"尝试保存图表到: {os.path.abspath(save_path)}")

    # 检查文件是否存在
    for key, file_path in files.items():
        if not os.path.exists(file_path):
            print(f"错误: 文件 {file_path} 不存在!")
            return

    # 加载数据并验证
    df_all = pd.read_csv(files['all'])
    print(f"全量数据形状: {df_all.shape}")
    print(f"全量数据中的算法: {df_all['alg'].unique()}")

    if not percentil:
        df_all = df_all[df_all['range'] != '400']
        print(f"过滤后的数据形状: {df_all.shape}")

    # 检查是否有该算法的数据
    df = df_all[df_all["alg"] == alg]
    if df.empty:
        print(f"错误: 在全量数据中找不到算法 {alg} 的记录")
        return

    # 创建全量数据的箱线图
    trace0 = go.Box(
        y=df[measure],
        x=df['range'],
        name="all",
        marker=dict(color='#1f78b4')
    )

    # 加载高/低范围数据
    df_high = pd.read_csv(files['high'])
    df_low = pd.read_csv(files['low'])
    print(f"高范围数据形状: {df_high.shape}")
    print(f"低范围数据形状: {df_low.shape}")

    # 检查是否有该算法的范围数据
    df_l = df_low[df_low["alg"] == alg]
    df_h = df_high[df_high["alg"] == alg]
    df_range = pd.concat([df_l, df_h])

    if df_range.empty:
        print(f"警告: 在范围数据中找不到算法 {alg} 的记录")
        trace1 = None
    else:
        trace1 = go.Box(
            y=df_range[measure],
            x=df_range['range'],
            name="range",
            marker=dict(color='#33a02c')
        )

    # 创建图表
    data = [trace0]
    if trace1 is not None:
        data.append(trace1)

    layout = go.Layout(
        yaxis=dict(title=measure, zeroline=False),
        xaxis=dict(title='Tg', zeroline=False),
        title=title,
        boxmode='group'
    )

    fig = go.Figure(data=data, layout=layout)

    # 尝试保存图表
    try:
        offline.plot(fig, filename=save_path, auto_open=False)
        print(f"图表已成功保存到 {save_path}")
    except Exception as e:
        print(f"保存图表时出错: {e}")


def plot_bar_algs(save_path, measure, percentil=False):
    """为所有算法创建箱线图，比较特定范围数据"""
    print(f"\n生成所有算法的范围箱线图 (percentil={percentil})")

    title = None
    if percentil:
        title = "All algorithms: Range (percentile)"
        save_path = f"{save_path}{measure}_all_algorithm_range_percentile.html"
        files = {
            'high': '../../result/aggr/result_high_percentil.csv',
            'low': '../../result/aggr/result_low_percentil.csv'
        }
    else:
        title = "All algorithms: Range"
        save_path = f"{save_path}{measure}_all_algorithm_range.html"
        files = {
            'high': '../../result/aggr/result_high.csv',
            'low': '../../result/aggr/result_low.csv'
        }

    # 打印绝对路径
    print(f"尝试保存图表到: {os.path.abspath(save_path)}")

    # 检查文件是否存在
    for key, file_path in files.items():
        if not os.path.exists(file_path):
            print(f"错误: 文件 {file_path} 不存在!")
            return

    # 加载数据并验证
    df_high = pd.read_csv(files['high'])
    df_low = pd.read_csv(files['low'])
    print(f"高范围数据形状: {df_high.shape}")
    print(f"低范围数据形状: {df_low.shape}")

    high_algs = df_high['alg'].unique() if not df_high.empty else []
    low_algs = df_low['alg'].unique() if not df_low.empty else []

    print(f"高范围数据中的算法: {high_algs}")
    print(f"低范围数据中的算法: {low_algs}")

    colors = ["#33a02c", "#fb9a99", "#a6cee3", "#7570b3"]
    algs = ["DT", "MLP", "SVM", "RF"]
    data = []

    for alg, col in zip(algs, colors):
        # 获取该算法在高/低范围的数据
        df_l = df_low[df_low["alg"] == alg]
        df_h = df_high[df_high["alg"] == alg]
        df = pd.concat([df_l, df_h])

        if df.empty:
            print(f"警告: 在范围数据中找不到算法 {alg} 的记录")
            continue

        trace = go.Box(
            y=df[measure],
            x=df['range'],
            name=alg,
            marker=dict(color=col)
        )
        data.append(trace)

    if not data:
        print("错误: 没有找到任何算法的范围数据!")
        return

    layout = go.Layout(
        yaxis=dict(title=measure, zeroline=False),
        xaxis=dict(title='Tg', zeroline=False),
        title=title,
        boxmode='group'
    )

    fig = go.Figure(data=data, layout=layout)

    # 尝试保存图表
    try:
        offline.plot(fig, filename=save_path, auto_open=False)
        print(f"图表已成功保存到 {save_path}")
    except Exception as e:
        print(f"保存图表时出错: {e}")


def plot_bar_all(save_path, measure, percentil=False):
    """为所有算法创建箱线图，比较全量数据"""
    print(f"\n生成所有算法的全量数据箱线图 (percentil={percentil})")

    title = None
    if percentil:
        title = "All algorithms: all data (percentile)"
        save_path = f"{save_path}{measure}_all_algorithm_percentile.html"
        file_path = '../../result/aggr/result_all_percentil.csv'
    else:
        title = "All algorithms: all data"
        save_path = f"{save_path}{measure}_all_algorithm.html"
        file_path = '../../result/aggr/result_all.csv'

    # 打印绝对路径
    print(f"尝试保存图表到: {os.path.abspath(save_path)}")

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在!")
        return

    # 加载数据并验证
    df_all = pd.read_csv(file_path)
    print(f"全量数据形状: {df_all.shape}")
    print(f"全量数据中的列名: {df_all.columns}")
    print(f"全量数据的前几行: \n{df_all.head()}")

    # 检查是否有 "all" 的记录（根据 1 列）
    all_data = df_all[df_all['1'] == 'all']
    if all_data.empty:
        print("警告: 数据中没有 1 列值为 'all' 的记录!")
        return
    else:
        print(f"alg 列值为 'all' 的数据形状: {all_data.shape}")
        print(f"alg 列值为 'all' 的数据中的范围: {all_data['range'].unique()}")

    # 绘图
    colors = ["#33a02c", "#fb9a99", "#a6cee3", "#7570b3"]
    algs = ["DT", "MLP", "SVM", "RF"]
    data = []

    for alg, col in zip(algs, colors):
        # 获取该算法在 "all" 范围的数据
        df = all_data[all_data["alg"] == alg]

        if df.empty:
            print(f"警告: 在全量数据中找不到算法 {alg} 且 alg 列值为 'all' 的记录")
            continue

        trace = go.Box(
            y=df[measure],
            x=df['alg'],
            name=alg,
            marker=dict(color=col)
        )
        data.append(trace)

    if not data:
        print("错误: 没有找到任何算法的全量数据!")
        return

    layout = go.Layout(
        yaxis=dict(title=measure, zeroline=False),
        xaxis=dict(title='Algorithm', zeroline=False),
        title=title,
        boxmode='group'
    )

    fig = go.Figure(data=data, layout=layout)

    # 尝试保存图表
    try:
        offline.plot(fig, filename=save_path, auto_open=False)
        print(f"图表已成功保存到 {save_path}")
    except Exception as e:
        print(f"保存图表时出错: {e}")


def run():
    # 获取并打印当前工作目录
    current_dir = os.getcwd()
    print(f"当前工作目录: {current_dir}")

    result_path = "../../result/aggr/"
    result_abs_path = os.path.abspath(result_path)
    print(f"聚合数据路径: {result_abs_path}")

    if not os.path.exists(result_path):
        os.makedirs(result_path)
        print(f"创建目录: {result_abs_path}")
    else:
        print(f"目录已存在: {result_abs_path}")
        # 列出目录内容
        print(f"目录内容: {os.listdir(result_abs_path)}")

    # 聚合数据
    print("\n开始聚合数据...")
    aggr("../../result/aggr/result_low/", "../../result/aggr/result_low.csv")
    aggr("../../result/aggr/result_high/", "../../result/aggr/result_high.csv")
    aggr("../../result/aggr/result_low_percentil/", "../../result/aggr/result_low_percentil.csv")
    aggr("../../result/aggr/result_high_percentil/", "../../result/aggr/result_high_percentil.csv")
    aggr_all("../../result/aggr/result_all/", "../../result/aggr/result_all.csv",
             "../../data/clean/oxides_Tg_train.csv", "Tg")
    aggr_all("../../result/aggr/result_all_percentil/", "../../result/aggr/result_all_percentil.csv",
             "../../data/clean/oxides_Tg_train.csv", "Tg")

    # 创建图表目录
    save_path = "../../result/plots/boxplot-range/"
    save_abs_path = os.path.abspath(save_path)
    print(f"\n图表保存路径: {save_abs_path}")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"创建目录: {save_abs_path}")
    else:
        print(f"目录已存在: {save_abs_path}")

    algs = ["DT", "MLP", "SVM", "RF"]
    measure = ["RMSE"]  # 只使用有效的度量

    print("\n开始生成图表...")
    for alg in algs:
        for me in measure:
            plot_bar(alg, save_path, me, True)
            plot_bar(alg, save_path, me, False)

    plot_bar_algs(save_path, measure[0], percentil=False)
    plot_bar_algs(save_path, measure[0], percentil=True)
    plot_bar_all(save_path, measure[0], percentil=False)
    plot_bar_all(save_path, measure[0], percentil=True)


if __name__ == "__main__":
    run()