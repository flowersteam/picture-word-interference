def dist_comparison(REF, O, N, distNames, probas=False):
    for pred_category in ["Superordinate", "Basic"]:
        for wa_category in ["Superordinate", "Basic"]:

            print("\n---")
            print(f"{pred_category.upper()} LABELS | {wa_category.upper()} wa")

            # --- REF / O
            u, p = mannwhitneyu(REF[pred_category][wa_category] if not probas else REF[pred_category],
                                O[pred_category][wa_category], alternative="two-sided")
            m1, m2 = statistics.median(
                REF[pred_category][wa_category] if not probas else REF[pred_category]), statistics.median(
                O[pred_category][wa_category])
            res = ""
            if p < 0.05:
                res = distNames[0] + " STATISTICALLY DIFFERENT FROM " + distNames[1] + " | "
                if m1 < m2:
                    res += distNames[0] + " < " + distNames[1]
                elif m1 > m2:
                    res += distNames[0] + " > " + distNames[1]
            else:
                res = distNames[0] + " NOT SIGNIFICANTLY DIFFERENT FROM " + distNames[1]
            print(res + " | u = " + str(u) + ",\t\t p = " + str(p))

            # --- REF / N
            u, p = mannwhitneyu(REF[pred_category][wa_category] if not probas else REF[pred_category],
                                N[pred_category][wa_category], alternative="two-sided")
            m1, m2 = statistics.median(
                REF[pred_category][wa_category] if not probas else REF[pred_category]), statistics.median(
                N[pred_category][wa_category])
            res = ""
            if p < 0.05:
                res = distNames[0] + " STATISTICALLY DIFFERENT FROM " + distNames[2] + " | "
                if m1 < m2:
                    res += distNames[0] + " < " + distNames[2]
                elif m1 > m2:
                    res += distNames[0] + " > " + distNames[2]
            else:
                res = distNames[0] + " NOT SIGNIFICANTLY DIFFERENT FROM " + distNames[2]
            print(res + " | u = " + str(u) + ",\t\t p = " + str(p))

            # --- O / N
            u, p = mannwhitneyu(O[pred_category][wa_category], N[pred_category][wa_category], alternative="two-sided")
            m1, m2 = statistics.median(O[pred_category][wa_category]), statistics.median(N[pred_category][wa_category])
            res = ""
            if p < 0.05:
                res = distNames[1] + " STATISTICALLY DIFFERENT FROM " + distNames[2] + " | "
                if m1 < m2:
                    res += distNames[1] + " < " + distNames[2]
                elif m1 > m2:
                    res += distNames[1] + " > " + distNames[2]
            else:
                res = distNames[1] + " NOT SIGNIFICANTLY DIFFERENT FROM " + distNames[2]
            print(res + " | u = " + str(u) + ",\t\t p = " + str(p))


def numerical_normalityCheck(REF, method):
    print("USING " + ("Shapiro-Wilk Test" if method == "Shapiro" else "D’Agostino’s K² Test"))
    for pred_category in ["Superordinate", "Basic"]:
        for wa_category in ["Superordinate", "Basic"]:
            print("---")
            print(f"REF FOR : {pred_category.upper()} LABELS | {wa_category.upper()} wa")
            if method == "Shapiro":
                _, p1 = shapiro(
                    REF[pred_category][wa_category] if testCategory != "PROBABILITY" else REF[pred_category])
                print(("  NOT NORMAL " if p1 < 0.05 else ("NORMAL     ")) + "\t p = " + str(p1))
            else:
                _, p2 = normaltest(
                    REF[pred_category][wa_category] if testCategory != "PROBABILITY" else REF[pred_category])
                print(("  NOT NORMAL " if p2 < 0.05 else ("NORMAL     ")) + "\t p = " + str(p2))
