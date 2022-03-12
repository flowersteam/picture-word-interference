import torch
from PIL import Image, ImageFont, ImageDraw
import numpy
import math
import os
import clip
import gensim
import matplotlib as plt

# SCRIPTS
model_w2v = gensim.models.KeyedVectors.load_word2vec_format("DATASETS/GoogleNews-vectors-negative300.bin", binary=True)


def get_stimuli(i, images, preprocess=None, word=None, raw=False):
    """
    Cell  #0 of Classification for word-superimposed images / UTILS
    :param i:
    :param images:
    :param preprocess:
    :param word:
    :param raw:
    :return:
    """
    img = Image.fromarray(images[i])
    #img.show()
    # --- ADD WORD TO STIMULI
    if None is not word:
        img_width, img_height = img.size
        fontsize = math.floor(img_width / 6)
        font = ImageFont.truetype("FONTS/arial.ttf", fontsize)
        text_width, text_height = font.getsize(word)
        x, y = img_width / 2 - text_width / 2, img_height / 2 - text_height / 2
        draw = ImageDraw.Draw(img)
        draw.text((x, y), word, (255, 0, 0), font=font, stroke_width=math.ceil(fontsize / 20),
                  stroke_fill=(255, 255, 255))
    # --- APPLY PREPROCESSING
    if None is not preprocess:
        return preprocess(img)
    elif not raw:
        return torch.tensor(numpy.array(img))
    else:
        return img


def get_stimulis(a, b, images, preprocess=None, word=None):
    """
        Cell  #1 of Classification for word-superimposed images / UTILS
    :param a:
    :param b:
    :param preprocess:
    :param word:
    :return:
    """
    return torch.cat(([get_stimuli(i, images, preprocess=preprocess, word=word).unsqueeze(0) for i in range(a, b)]))


def compute_original_preds(device, model, preprocess, images, text_basic, text_super, model_name, contexts, context, batch_size=128):
    """
            Cell #2 of Classification for word-superimposed images / UTILS
    :param model:
    :param preprocess:
    :param text_basic:
    :param text_super:
    :param model_name:
    :param contexts:
    :param context:
    :param device:
    :param batch_size:
    :return:
    """
    data_path = "DATA/" + model_name + "_" + "context" + str(contexts.index(context)) + "_original_preds.pt"
    dataset_size = 274

    # ----- CHECKING IF DATA ALREADY EXISTS -------------------------------------------
    if os.path.exists(data_path):
        original_predictions = torch.load(data_path, map_location=device)
    else:
        original_predictions = {}
        # ---- PROCESSING DATA IN MINI-BATCHES --------------------------------
        for i in range(0, math.ceil(dataset_size / batch_size)):
            a, b = i * batch_size, min(dataset_size, i * batch_size + batch_size)
            batch = get_stimulis(a, b, images, preprocess).to(device)

            # --- BASIC PREDICTIONS ------------------------------------------
            with torch.no_grad():
                logits_per_image_basic, _ = model(batch, text_basic)
                probs_basic = logits_per_image_basic.softmax(dim=-1)
                _, preds_basic = torch.max(probs_basic, 1)

            # --- SUPERORDINATE PREDICTIONS ----------------------------------
            with torch.no_grad():
                logits_per_image_super, _ = model(batch, text_super)
                probs_super = logits_per_image_super.softmax(dim=-1)
                _, preds_super = torch.max(probs_super, 1)

            # --- SAVING PREDICTIONS -------------------------------------------
            for j in range(a, b):
                result = {"Superordinate": {}, "Basic": {}}

                result["Superordinate"]["Prediction"] = preds_super[j - a]
                result["Superordinate"]["Logits"] = logits_per_image_super[j - a]
                result["Superordinate"]["Probas"] = probs_super[j - a]

                result["Basic"]["Prediction"] = preds_basic[j - a]
                result["Basic"]["Logits"] = logits_per_image_basic[j - a]
                result["Basic"]["Probas"] = probs_basic[j - a]

                original_predictions[j] = result

        torch.save(original_predictions, data_path)
    return original_predictions


def get_original_preds(i, original_predictions, super_labels, basic_labels, display=False, images=[]):
    """
        Cell #3 of Classification for word-superimposed images / UTILS
    :param i:
    :param original_predictions:
    :param super_labels:
    :param basic_labels:
    :param display:
    :return:
    """
    data = original_predictions[i]

    super_pred = data["Superordinate"]["Prediction"]
    super_logit = data["Superordinate"]["Logits"][super_pred]
    super_proba = data["Superordinate"]["Probas"][super_pred]

    basic_pred = data["Basic"]["Prediction"]
    basic_logit = data["Basic"]["Logits"][basic_pred]
    basic_proba = data["Basic"]["Probas"][basic_pred]

    if display:
        stimuli = get_stimuli(i, images)
        plt.imshow(stimuli.cpu())
        plt.show()
        print(f"SUPERORDINATE LABEL \t: {super_labels[super_pred]} | {super_proba * 100}%")
        print(f"BASIC LABEL \t\t: {basic_labels[basic_pred]} | {basic_proba * 100}%")

    return data


def compute_new_preds(device, model, preprocess, images, super_labels, basic_labels, text_basic, text_super, model_name, contexts, context, batch_size=128):
    """
            Cell #4 of Classification for word-superimposed images / UTILS
    :param model_name:
    :param model:
    :param preprocess:
    :param super_labels:
    :param basic_labels:
    :param text_basic:
    :param text_super:
    :param device:
    :param contexts:
    :param context:
    :param batch_size:
    :return:
    """
    data_path = "DATA/" + model_name + "_" + "context" + str(contexts.index(context)) + "_wordsAdd_preds.pt"
    dataset_size = 274
    words = list(set(super_labels) | set(basic_labels))

    print(len(words))

    # ----- CHECKING IF DATA ALREADY EXISTS -------------------------------------------
    if os.path.exists(data_path):
        wordsAdd_predictions = torch.load(data_path, map_location=device)
    else:
        wordsAdd_predictions = {}
    start = len(wordsAdd_predictions.keys())  # Start at where we're at

    # ----- COLLECTING THE PREDICTIONS FOR EACH WORD-IMAGE PAIRS -------------------
    for w in range(start, len(words)):
        word = words[w]
        wordsAdd_predictions[word] = []
        print(word)
        print(f"{w + 1}/{len(words)}")

        # ---- PROCESSING DATA IN MINI-BATCHES --------------------------------
        for i in range(0, math.ceil(dataset_size / batch_size)):
            a, b = i * batch_size, min(dataset_size, i * batch_size + batch_size)
            batch = get_stimulis(a, b, images, preprocess, word).to(device)

            # --- BASIC PREDICTIONS ------------------------------------------
            with torch.no_grad():
                logits_per_image_basic, _ = model(batch, text_basic)
                probs_basic = logits_per_image_basic.softmax(dim=-1)
                _, preds_basic = torch.max(probs_basic, 1)

            # --- SUPERORDINATE PREDICTIONS ----------------------------------
            with torch.no_grad():
                logits_per_image_super, _ = model(batch, text_super)
                probs_super = logits_per_image_super.softmax(dim=-1)
                _, preds_super = torch.max(probs_super, 1)

            # --- SAVING PREDICTIONS -------------------------------------------
            for j in range(a, b):
                result = {"Superordinate": {}, "Basic": {}}

                result["Superordinate"]["Prediction"] = preds_super[j - a]
                result["Superordinate"]["Logits"] = logits_per_image_super[j - a]
                result["Superordinate"]["Probas"] = probs_super[j - a]

                result["Basic"]["Prediction"] = preds_basic[j - a]
                result["Basic"]["Logits"] = logits_per_image_basic[j - a]
                result["Basic"]["Probas"] = probs_basic[j - a]

                wordsAdd_predictions[word].append(result)

    torch.save(wordsAdd_predictions, data_path)
    return wordsAdd_predictions


def get_wordAdd_preds(i, word, wordsAdd_predictions, super_labels, basic_labels, display=False, original_predictions=[], images=[]):
    """
        Cell #5 of Classification for word-superimposed images / UTILS
    :param i:
    :param word:
    :param wordsAdd_predictions:
    :param super_labels:
    :param basic_labels:
    :param display:
    :return:
    """
    data = wordsAdd_predictions[word][i]

    if display:
        og_data = get_original_preds(i, original_predictions, super_labels, basic_labels)
        og_super_pred = og_data["Superordinate"]["Prediction"]
        og_super_logit = og_data["Superordinate"]["Logits"][og_super_pred]
        og_super_proba = og_data["Superordinate"]["Probas"][og_super_pred]

        super_pred = data["Superordinate"]["Prediction"]
        super_logit = data["Superordinate"]["Logits"][super_pred]
        super_proba = data["Superordinate"]["Probas"][super_pred]

        og_super_newLogit = data["Superordinate"]["Logits"][og_super_pred]
        og_super_newProba = data["Superordinate"]["Probas"][og_super_pred]

        # --------------------------------------------------------------

        og_basic_pred = og_data["Basic"]["Prediction"]
        og_basic_logit = og_data["Basic"]["Logits"][og_basic_pred]
        og_basic_proba = og_data["Basic"]["Probas"][og_basic_pred]

        basic_pred = data["Basic"]["Prediction"]
        basic_logit = data["Basic"]["Logits"][basic_pred]
        basic_proba = data["Basic"]["Probas"][basic_pred]

        og_basic_newLogit = data["Basic"]["Logits"][og_basic_pred]
        og_basic_newProba = data["Basic"]["Probas"][og_basic_pred]

        stimuli = get_stimuli(i, images, preprocess=None, word=word)
        plt.imshow(stimuli.cpu())
        plt.show()

        print(f"SUPERORDINATE \t NEW \t\t: {super_labels[super_pred]} | {super_proba * 100}%")
        print(
            f"\t\t ORIGINAL \t: {super_labels[og_super_pred]} | {og_super_proba * 100}% --> {og_super_newProba * 100}%")
        print(
            "----------------------------------------------------------------------------------------------------------------------------------------")
        print(f"BASIC \t\t NEW \t\t: {basic_labels[basic_pred]} | {basic_proba * 100}%")
        print(
            f"\t\t ORIGINAL \t: {basic_labels[og_basic_pred]} | {og_basic_proba * 100}% --> {og_basic_newProba * 100}%")

    return data


def semantic_similarity_w2v(w1, w2, model_w2v=model_w2v):
    f1, f2 = torch.tensor(model_w2v[w1]), torch.tensor(model_w2v[w2])
    return torch.nn.CosineSimilarity(dim=0)(f1, f2).item()


def get_switching_rate(super_labels, basic_labels, original_predictions, wordsAdd_predictions):
    miss_rates = {"Superordinate": {}, "Basic": {}}
    miss_rates["Superordinate"]["Superordinate"] = 0.0
    miss_rates["Basic"]["Superordinate"] = 0.0
    miss_rates["Superordinate"]["Basic"] = 0.0
    miss_rates["Basic"]["Basic"] = 0.0

    # ----- SUPERORDINATE WA ------------------------------------------------------------------------------------------------------------------------
    counted_basic = 0
    counted_super = 0
    for wa in super_labels:
        for i in range(0, 274):
            original_preds, new_preds = get_original_preds(i, original_predictions, super_labels, basic_labels), get_wordAdd_preds(i, wa, wordsAdd_predictions, super_labels, basic_labels)
            if (wa != super_labels[original_preds["Superordinate"]["Prediction"]]):
                miss_rates["Superordinate"]["Superordinate"] += (
                        original_preds["Superordinate"]["Prediction"].item() != new_preds["Superordinate"][
                    "Prediction"].item())
                miss_rates["Basic"]["Superordinate"] += (
                        original_preds["Basic"]["Prediction"].item() != new_preds["Basic"]["Prediction"].item())
                counted_super += 1

    miss_rates["Superordinate"]["Superordinate"] /= counted_super
    miss_rates["Basic"]["Superordinate"] /= counted_super

    # ----- BASIC WA --------------------------------------------------------------------------------------------------------------------------------
    counted_basic = 0
    for wa in basic_labels:
        for i in range(0, 274):
            original_preds, new_preds = get_original_preds(i, original_predictions, super_labels, basic_labels), get_wordAdd_preds(i, wa, wordsAdd_predictions, super_labels, basic_labels)
            if (wa != basic_labels[original_preds["Basic"]["Prediction"]]):
                miss_rates["Basic"]["Basic"] += (
                        original_preds["Basic"]["Prediction"].item() != new_preds["Basic"]["Prediction"].item())
                miss_rates["Superordinate"]["Basic"] += (
                        original_preds["Superordinate"]["Prediction"].item() != new_preds["Superordinate"][
                    "Prediction"].item())
                counted_basic += 1

    miss_rates["Superordinate"]["Basic"] /= counted_basic
    miss_rates["Basic"]["Basic"] /= counted_basic

    return miss_rates


def get_word_correlation_references(metric):
    references = {"Superordinate": {}, "Basic": {}}
    references["Superordinate"]["Superordinate"] = []
    references["Basic"]["Superordinate"] = []
    references["Superordinate"]["Basic"] = []
    references["Basic"]["Basic"] = []

    for pred_category in ["Basic", "Superordinate"]:
        for wa_category in ["Basic", "Superordinate"]:
            added_words = (super_labels if wa_category == "Superordinate" else basic_labels)
            labels = (super_labels if pred_category == "Superordinate" else basic_labels)

            similarities = []
            for wa in added_words:
                for i in range(0, 274):
                    original_data = get_original_preds(i, original_predictions, super_labels, basic_labels)
                    original_pred = original_data[pred_category]["Prediction"]
                    original_label = labels[original_pred]

                    # print(wa)
                    similarities.append(metric(wa, original_label))

            references[pred_category][wa_category] = similarities

    return references


def get_word_correlation_references_nonswitchonly(metric, basic_labels, super_labels, original_predictions, wordsAdd_predictions):
    references = {"Superordinate": {}, "Basic": {}}
    references["Superordinate"]["Superordinate"] = []
    references["Basic"]["Superordinate"] = []
    references["Superordinate"]["Basic"] = []
    references["Basic"]["Basic"] = []

    for pred_category in ["Basic", "Superordinate"]:
        for wa_category in ["Basic", "Superordinate"]:
            added_words = (super_labels if wa_category == "Superordinate" else basic_labels)
            labels = (super_labels if pred_category == "Superordinate" else basic_labels)

            similarities = []
            for wa in added_words:
                for i in range(0, 274):
                    original_data, new_data = get_original_preds(i, original_predictions, super_labels, basic_labels), get_wordAdd_preds(i, wa, wordsAdd_predictions, super_labels, basic_labels)
                    original_pred, new_pred = original_data[pred_category]["Prediction"], new_data[pred_category][
                        "Prediction"]
                    original_label, new_label = labels[original_pred], labels[new_pred]

                    if original_pred == new_pred:
                        similarities.append(metric(wa, original_label))

            references[pred_category][wa_category] = similarities

    return references


def get_OAC(metric, basic_labels, super_labels, original_predictions, wordsAdd_predictions):
    references = {"Superordinate": {}, "Basic": {}}
    references["Superordinate"]["Superordinate"] = []
    references["Basic"]["Superordinate"] = []
    references["Superordinate"]["Basic"] = []
    references["Basic"]["Basic"] = []

    for pred_category in ["Basic", "Superordinate"]:
        for wa_category in ["Basic", "Superordinate"]:
            added_words = (super_labels if wa_category == "Superordinate" else basic_labels)
            labels = (super_labels if pred_category == "Superordinate" else basic_labels)

            similarities = []
            for wa in added_words:
                for i in range(0, 274):
                    original_data, new_data = get_original_preds(i, original_predictions, super_labels, basic_labels), get_wordAdd_preds(i, wa, wordsAdd_predictions, super_labels, basic_labels)
                    original_pred, new_pred = original_data[pred_category]["Prediction"], new_data[pred_category][
                        "Prediction"]
                    original_label, new_label = labels[original_pred], labels[new_pred]

                    if original_pred != new_pred:
                        similarities.append(metric(wa, original_label))

            references[pred_category][wa_category] = similarities

    return references


def get_NAC(metric):
    references = {"Superordinate": {}, "Basic": {}}
    references["Superordinate"]["Superordinate"] = []
    references["Basic"]["Superordinate"] = []
    references["Superordinate"]["Basic"] = []
    references["Basic"]["Basic"] = []

    for pred_category in ["Basic", "Superordinate"]:
        for wa_category in ["Basic", "Superordinate"]:
            added_words = (super_labels if wa_category == "Superordinate" else basic_labels)
            labels = (super_labels if pred_category == "Superordinate" else basic_labels)

            similarities = []
            for wa in added_words:
                for i in range(0, 274):
                    original_data, new_data = get_original_preds(i), get_wordAdd_preds(i, wa)
                    original_pred, new_pred = original_data[pred_category]["Prediction"], new_data[pred_category][
                        "Prediction"]
                    original_label, new_label = labels[original_pred], labels[new_pred]

                    if original_pred != new_pred:
                        similarities.append(metric(wa, new_label))

            references[pred_category][wa_category] = similarities

    return references


def get_probabilities_references():
    references = {"Basic": [], "Superordinate": []}

    for i in range(0, 274):
        original_preds = get_original_preds(i)
        super_pred, basic_pred = original_preds["Superordinate"]["Prediction"], original_preds["Basic"]["Prediction"]
        references["Superordinate"].append(original_preds["Superordinate"]["Probas"][super_pred].item())
        references["Basic"].append(original_preds["Basic"]["Probas"][basic_pred].item())

    return references


def get_probabilities_references_nonswitched(basic_labels, super_labels, original_predictions, wordsAdd_predictions):
    original_probs = {"Superordinate": {}, "Basic": {}}
    original_probs["Superordinate"]["Superordinate"] = []
    original_probs["Basic"]["Superordinate"] = []
    original_probs["Superordinate"]["Basic"] = []
    original_probs["Basic"]["Basic"] = []

    # ----- SUPERORDINATE WA -------------------------------------------------------------------------------------------------
    for wa in super_labels:
        for i in range(0, 274):
            original_preds, new_preds = get_original_preds(i, original_predictions, super_labels, basic_labels), get_wordAdd_preds(i, wa, wordsAdd_predictions, super_labels, basic_labels)

            # --- SUPERORDINATE PRED
            originalPred, newPred = original_preds["Superordinate"]["Prediction"], new_preds["Superordinate"][
                "Prediction"]
            if originalPred == newPred:
                original_probs["Superordinate"]["Superordinate"].append(
                    original_preds["Superordinate"]["Probas"][originalPred].item())

            # --- BASIC PRED
            originalPred, newPred = original_preds["Basic"]["Prediction"], new_preds["Basic"]["Prediction"]
            if originalPred == newPred:
                original_probs["Basic"]["Superordinate"].append(original_preds["Basic"]["Probas"][originalPred].item())

    # ----- BASIC WA --------------------------------------------------------------------------------------------------------
    for wa in basic_labels:
        for i in range(0, 274):
            original_preds, new_preds = get_original_preds(i, original_predictions, super_labels, basic_labels), get_wordAdd_preds(i, wa, wordsAdd_predictions, super_labels, basic_labels)

            # --- SUPERORDINATE PRED
            originalPred, newPred = original_preds["Superordinate"]["Prediction"], new_preds["Superordinate"][
                "Prediction"]
            if originalPred == newPred:
                original_probs["Superordinate"]["Basic"].append(
                    original_preds["Superordinate"]["Probas"][originalPred].item())

            # --- BASIC PRED
            originalPred, newPred = original_preds["Basic"]["Prediction"], new_preds["Basic"]["Prediction"]
            if originalPred == newPred:
                original_probs["Basic"]["Basic"].append(original_preds["Basic"]["Probas"][originalPred].item())

    return original_probs


def get_COM_original(basic_labels, super_labels, original_predictions, wordsAdd_predictions):
    original_probs = {"Superordinate": {}, "Basic": {}}
    original_probs["Superordinate"]["Superordinate"] = []
    original_probs["Basic"]["Superordinate"] = []
    original_probs["Superordinate"]["Basic"] = []
    original_probs["Basic"]["Basic"] = []

    # ----- SUPERORDINATE WA -------------------------------------------------------------------------------------------------
    for wa in super_labels:
        for i in range(0, 274):
            original_preds, new_preds = get_original_preds(i, original_predictions, super_labels, basic_labels), get_wordAdd_preds(i, wa, wordsAdd_predictions, super_labels, basic_labels)

            # --- SUPERORDINATE PRED
            originalPred, newPred = original_preds["Superordinate"]["Prediction"], new_preds["Superordinate"][
                "Prediction"]
            if originalPred != newPred:
                original_probs["Superordinate"]["Superordinate"].append(
                    original_preds["Superordinate"]["Probas"][originalPred].item())

            # --- BASIC PRED
            originalPred, newPred = original_preds["Basic"]["Prediction"], new_preds["Basic"]["Prediction"]
            if originalPred != newPred:
                original_probs["Basic"]["Superordinate"].append(original_preds["Basic"]["Probas"][originalPred].item())

    # ----- BASIC WA --------------------------------------------------------------------------------------------------------
    for wa in basic_labels:
        for i in range(0, 274):
            original_preds, new_preds = get_original_preds(i, original_predictions, super_labels, basic_labels), get_wordAdd_preds(i, wa, wordsAdd_predictions, super_labels, basic_labels)

            # --- SUPERORDINATE PRED
            originalPred, newPred = original_preds["Superordinate"]["Prediction"], new_preds["Superordinate"][
                "Prediction"]
            if originalPred != newPred:
                original_probs["Superordinate"]["Basic"].append(
                    original_preds["Superordinate"]["Probas"][originalPred].item())

            # --- BASIC PRED
            originalPred, newPred = original_preds["Basic"]["Prediction"], new_preds["Basic"]["Prediction"]
            if originalPred != newPred:
                original_probs["Basic"]["Basic"].append(original_preds["Basic"]["Probas"][originalPred].item())

    return original_probs


def get_COM_new(basic_labels, super_labels, original_predictions, wordsAdd_predictions):
    new_probs = {"Superordinate": {}, "Basic": {}}
    new_probs["Superordinate"]["Superordinate"] = []
    new_probs["Basic"]["Superordinate"] = []
    new_probs["Superordinate"]["Basic"] = []
    new_probs["Basic"]["Basic"] = []

    # ----- SUPERORDINATE WA -------------------------------------------------------------------------------------------------
    for wa in super_labels:
        for i in range(0, 274):
            original_preds, new_preds = get_original_preds(i, original_predictions, super_labels, basic_labels), get_wordAdd_preds(i, wa, wordsAdd_predictions, super_labels, basic_labels)

            # --- SUPERORDINATE PRED
            originalPred, newPred = original_preds["Superordinate"]["Prediction"], new_preds["Superordinate"][
                "Prediction"]
            if originalPred != newPred:
                new_probs["Superordinate"]["Superordinate"].append(new_preds["Superordinate"]["Probas"][newPred].item())

            # --- BASIC PRED
            originalPred, newPred = original_preds["Basic"]["Prediction"], new_preds["Basic"]["Prediction"]
            if originalPred != newPred:
                new_probs["Basic"]["Superordinate"].append(new_preds["Basic"]["Probas"][newPred].item())

    # ----- BASIC WA --------------------------------------------------------------------------------------------------------
    for wa in basic_labels:
        for i in range(0, 274):
            original_preds, new_preds = get_original_preds(i, original_predictions, super_labels, basic_labels), get_wordAdd_preds(i, wa, wordsAdd_predictions, super_labels, basic_labels)

            # --- SUPERORDINATE PRED
            originalPred, newPred = original_preds["Superordinate"]["Prediction"], new_preds["Superordinate"][
                "Prediction"]
            if originalPred != newPred:
                new_probs["Superordinate"]["Basic"].append(new_preds["Superordinate"]["Probas"][newPred].item())

            # --- BASIC PRED
            originalPred, newPred = original_preds["Basic"]["Prediction"], new_preds["Basic"]["Prediction"]
            if originalPred != newPred:
                new_probs["Basic"]["Basic"].append(new_preds["Basic"]["Probas"][newPred].item())

    return new_probs


def get_COM_neworiginal(basic_labels, super_labels, original_predictions, wordsAdd_predictions):
    new_probs = {"Superordinate": {}, "Basic": {}}
    new_probs["Superordinate"]["Superordinate"] = []
    new_probs["Basic"]["Superordinate"] = []
    new_probs["Superordinate"]["Basic"] = []
    new_probs["Basic"]["Basic"] = []

    # ----- SUPERORDINATE WA -------------------------------------------------------------------------------------------------
    for wa in super_labels:
        for i in range(0, 274):
            original_preds, new_preds = get_original_preds(i, original_predictions, super_labels, basic_labels), get_wordAdd_preds(i, wa, wordsAdd_predictions, super_labels, basic_labels)

            # --- SUPERORDINATE PRED
            originalPred, newPred = original_preds["Superordinate"]["Prediction"], new_preds["Superordinate"][
                "Prediction"]
            if originalPred != newPred:
                new_probs["Superordinate"]["Superordinate"].append(
                    new_preds["Superordinate"]["Probas"][originalPred].item())

            # --- BASIC PRED
            originalPred, newPred = original_preds["Basic"]["Prediction"], new_preds["Basic"]["Prediction"]
            if originalPred != newPred:
                new_probs["Basic"]["Superordinate"].append(new_preds["Basic"]["Probas"][originalPred].item())

    # ----- BASIC WA --------------------------------------------------------------------------------------------------------
    for wa in basic_labels:
        for i in range(0, 274):
            original_preds, new_preds = get_original_preds(i, original_predictions, super_labels, basic_labels), get_wordAdd_preds(i, wa, wordsAdd_predictions, super_labels, basic_labels)

            # --- SUPERORDINATE PRED
            originalPred, newPred = original_preds["Superordinate"]["Prediction"], new_preds["Superordinate"][
                "Prediction"]
            if originalPred != newPred:
                new_probs["Superordinate"]["Basic"].append(new_preds["Superordinate"]["Probas"][originalPred].item())

            # --- BASIC PRED
            originalPred, newPred = original_preds["Basic"]["Prediction"], new_preds["Basic"]["Prediction"]
            if originalPred != newPred:
                new_probs["Basic"]["Basic"].append(new_preds["Basic"]["Probas"][originalPred].item())

    return new_probs

