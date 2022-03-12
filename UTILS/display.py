import matplotlib.pyplot as plt


def display_test_boxplots(REF, O, N, testCategory, distNames):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    i, j = 0, 0
    for pred_category in ["Superordinate", "Basic"]:
        for wa_category in ["Superordinate", "Basic"]:
            box_ref = display_boxes(axs[i][j], [
                REF[pred_category][wa_category] if testCategory != "PROBABILITY" else REF[pred_category], [], []],
                                    [distNames[0], distNames[1], distNames[2]], "", testCategory, "", (-0.1, 1.1))
            box_oac = display_boxes(axs[i][j], [[], O[pred_category][wa_category], []],
                                    [distNames[0], distNames[1], distNames[2]], "", testCategory, "", (-0.1, 1.1))
            box_nac = display_boxes(axs[i][j], [[], [], N[pred_category][wa_category]],
                                    [distNames[0], distNames[1], distNames[2]], "", testCategory, "", (-0.1, 1.1))
            axs[i][j].set_title(f"{pred_category.upper()} LABELS | {wa_category.upper()} wa")

            set_box_colors(box_ref, ["white", "black", "black", "black"])
            set_box_colors(box_oac, ["black", "green", "green", "green"])
            set_box_colors(box_nac, ["black", "red", "red", "red"])
            j += 1
        i += 1
        j = 0
        fig.subplots_adjust(wspace=0.25, hspace=0.3)


def display_test_boxplots_twoconds(REF, O, testCategory, distNames):
    # I changed the order of panels (pred_category and wa_category)to make consistency in the paper
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    i, j = 0, 0
    for wa_category in ["Superordinate", "Basic"]:
        for pred_category in ["Superordinate", "Basic"]:
            # exclude the same category between images and words
            tmp = REF[pred_category][wa_category] if testCategory != "PROBABILITY" else REF[pred_category]
            tmp = [i for i in tmp if i != 1.0]
            tmp = [tmp, []]
            box_ref = display_boxes_forpaper(axs[i][j], tmp, [distNames[0], distNames[1]], "", testCategory, "",
                                             (-0.1, 1.1))
            box_oac = display_boxes_forpaper(axs[i][j], [[], O[pred_category][wa_category]],
                                             [distNames[0], distNames[1]], "", testCategory, "", (-0.1, 1.1))
            # axs[i][j].set_title(f"{pred_category.upper()} LABELS | {wa_category.upper()} ew")

            # save_csv_two(REF[pred_category][wa_category] if testCategory!="PROBABILITY" else REF[pred_category],O[pred_category][wa_category],pred_category,wa_category,testCategory)

            set_box_colors(box_ref, ["white", "black", "black", "black"])
            set_box_colors(box_oac, ["black", "red", "red", "red"])
            j += 1
        i += 1
        j = 0
        fig.subplots_adjust(wspace=0.25, hspace=0.3)


def display_test_boxplots_fourconds(REF, O, ON, N, testCategory, distNames):
    # I changed the order of panels (pred_category and wa_category)to make consistency in the paper
    fig, axs = plt.subplots(2, 2, figsize=(24, 10))
    i, j = 0, 0
    for wa_category in ["Superordinate", "Basic"]:
        for pred_category in ["Superordinate", "Basic"]:
            # exclude the same category between images and words
            tmp = [REF[pred_category][wa_category], [], [], []]
            # tmp = [i for i in tmp if i != 1.0]
            # tmp = [tmp,[]]
            box_ref = display_boxes_forpaper(axs[i][j], tmp, [distNames[0], distNames[1], distNames[2], distNames[3]],
                                             "", testCategory, "", (-0.1, 1.1))
            box_switchedoriginal = display_boxes_forpaper(axs[i][j], [[], O[pred_category][wa_category], [], []],
                                                          [distNames[0], distNames[1], distNames[2], distNames[3]], "",
                                                          testCategory, "", (-0.1, 1.1))
            box_oridinalnew = display_boxes_forpaper(axs[i][j], [[], [], ON[pred_category][wa_category], []],
                                                     [distNames[0], distNames[1], distNames[2], distNames[3]], "",
                                                     testCategory, "", (-0.1, 1.1))
            box_newlabel = display_boxes_forpaper(axs[i][j], [[], [], [], N[pred_category][wa_category]],
                                                  [distNames[0], distNames[1], distNames[2], distNames[3]], "",
                                                  testCategory, "", (-0.1, 1.1))
            # axs[i][j].set_title(f"{pred_category.upper()} LABELS | {wa_category.upper()} ew")

            # save_csv_four(REF[pred_category][wa_category],O[pred_category][wa_category],ON[pred_category][wa_category],N[pred_category][wa_category],pred_category,wa_category,testCategory)

            set_box_colors(box_ref, ["white", "black", "black", "black"])
            set_box_colors(box_switchedoriginal, ["black", "red", "red", "red"])
            set_box_colors(box_oridinalnew, ["black", "red", "red", "red"])
            set_box_colors(box_newlabel, ["black", "blue", "blue", "blue"])
            j += 1
        i += 1
        j = 0
    fig.subplots_adjust(wspace=0.25, hspace=0.3)


def display_boxes_forpaper(ax, data, tickLabels, xlabel, ylabel, title, yscale):
    boxes = ax.boxplot(data, showfliers=True, patch_artist=True, widths=0.5)
    ax.set_ylabel(ylabel, labelpad=20)
    ax.set_xlabel(xlabel, labelpad=20)
    ax.set_xticklabels(tickLabels, fontsize=20)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=18)
    ax.set_title(title)
    ax.set_ylim(yscale)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # plt.tight_layout()

    return boxes


def display_test_histograms(REF, O, N, testCategory, distNames, bins):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    i, j = 0, 0
    for pred_category in ["Superordinate", "Basic"]:
        for wa_category in ["Superordinate", "Basic"]:
            axs[i][j].set_title(f"{pred_category.upper()} LABELS | {wa_category.upper()} wa")
            axs[i][j].set_xlabel(testCategory)
            axs[i][j].hist(REF[pred_category][wa_category] if testCategory != "PROBABILITY" else REF[pred_category],
                           bins=bins, color="black", alpha=1, density=True, label=distNames[0])
            axs[i][j].hist(O[pred_category][wa_category], bins=bins, color="green", alpha=0.7, density=True,
                           label=distNames[1])
            axs[i][j].hist(N[pred_category][wa_category], bins=bins, color="red", alpha=0.7, density=True,
                           label=distNames[2])
            axs[i][j].set_ylabel("%")

            if (i, j) == (1, 1):
                handles, labels = axs[i][j].get_legend_handles_labels()
            j += 1
        i += 1
        j = 0
    fig.legend(handles, labels, loc=(0.91, 0.828))
    fig.subplots_adjust(wspace=0.3, hspace=0.3)


def visual_normalityCheck(REF):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    i, j = 0, 0

    for pred_category in ["Superordinate", "Basic"]:
        for wa_category in ["Superordinate", "Basic"]:
            scipy.stats.probplot(
                REF[pred_category][wa_category] if testCategory != "PROBABILITY" else REF[pred_category], dist="norm",
                plot=axs[i][j])
            axs[i][j].set_title(f"{pred_category.upper()} LABELS | {wa_category.upper()} wa")

            j += 1
        i += 1
        j = 0
    fig.subplots_adjust(wspace=0.3, hspace=0.3)


def display_boxes(ax, data, tickLabels, xlabel, ylabel, title, yscale):
    boxes = ax.boxplot(data, showfliers=False, patch_artist=True)
    ax.set_ylabel(ylabel, labelpad=15)
    ax.set_xlabel(xlabel, labelpad=15)
    ax.set_xticklabels(tickLabels, fontsize=15)
    ax.set_title(title)
    ax.set_ylim(yscale)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return boxes


def set_box_colors(box, colors):
    for element in box['medians']:
        element.set_color(colors[0])
        element.set_linewidth(2)
    for element in box['boxes']:
        element.set_facecolor(colors[1])
        element.set_linewidth(2)
    for element in box['whiskers']:
        element.set_linewidth(2)
    for element in box['caps']:
        element.set_linewidth(2)
