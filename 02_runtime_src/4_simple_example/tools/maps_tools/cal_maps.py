from matplotlib import pyplot as plt
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--result-path',
        type=str,
        required=True,
        help='eval and perf result file path')

    parser.add_argument(
        '--maps-result-path',
        type=str,
        default="maps_result.jpg",
        required=False,
        help='maps result save path')

    args = parser.parse_args()
    print(args)

    accuracy_array = []
    fps_array = []
    name_array = []
    with open(args.result_path) as lines:
        array = lines.readlines()
        count = 0
        for i in array:
            if count % 3 == 0:
                i = i.strip('\n')
                name_array.append(i)
            elif count % 3 == 1:
                i = i.strip('\n').split(":")[1]
                accuracy_array.append(float(i))
            elif count % 3 == 2:
                i = i.strip('\n').split(":")[1]
                fps_array.append(int(float(i)))
            count = count + 1

    y_1 = []
    x_1 = []
    x_2 = []
    y_2 = []
    index = 0
    name_array.reverse()
    # increase order
    fps_array.sort()
    # descending order
    accuracy_array.sort(reverse=True)
    min_fps = fps_array[0]
    max_fps = fps_array[len(fps_array) - 1]
    min_accuracy = accuracy_array[len(accuracy_array) - 1]
    max_accuracy = accuracy_array[0]

    # print(fps_array)
    # print(accuracy_array)
    for i in range(len(name_array)):
        if index == 0:
            y_1.append(accuracy_array[i])
            y_1.append(accuracy_array[i])
            x_1.append(0)
            x_1.append(fps_array[i])
            x_2.append(0)
        elif index == len(name_array) - 1:
            y_1.append(accuracy_array[i])
            y_1.append(accuracy_array[i])
            x_1.append(fps_array[i])
            x_1.append(fps_array[i])
            y_2.append(accuracy_array[i])
            y_2.append(accuracy_array[i])
            x_2.append(fps_array[i])
        else:
            y_1.append(accuracy_array[i])
            x_1.append(fps_array[i])
        index = index + 1

    # set style
    plt.style.use('seaborn-dark')

    # set figsize
    # plt.figure(figsize=(16, 8), dpi=80)
    plt.plot(x_1, y_1, color='#87CEEB', markersize=8.0, marker='o',
             linestyle='-', linewidth=3)
    plt.plot(x_2, y_2, color='#87CEEB', markersize=8.0, marker='o',
             linestyle='-', linewidth=3)

    # set x and y ax
    y_begin = round(min_accuracy, 2) - 0.01
    y_end = round(max_accuracy, 2) + 0.01
    x_end = (max_fps // 100 + 1) * 100
    plt.ylim(y_begin, y_end)
    plt.xlim((0, y_end))

    x_ticks = []
    for i in range(max_fps // 100 + 2):
        x_ticks.append(i * 100)
    plt.xticks(x_ticks)

    # draw area
    plt.fill_between(x_1[0:max_fps], min_accuracy, y_1, facecolor='#6495ED',
                     alpha=0.3)

    # set txt
    for i in range(len(accuracy_array)):
        plt.text(fps_array[i], accuracy_array[i], name_array[i], ha='center',
                 va='bottom', fontsize=12, color='#FFA500')

    # x y table
    plt.xlabel("FPS", fontsize=22)
    plt.ylabel("Top-1 Accuracy", fontsize=22)
    plt.title("ImageNet AFPS", fontsize=22)

    # show
    plt.savefig(args.maps_result_path)
    print("draw maps success!!")
