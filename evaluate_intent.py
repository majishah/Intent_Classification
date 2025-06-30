import json
import os
import logging
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import torch
import numpy as np # Needed for comparison

# -------------------
# Configuration (Same as before)
# -------------------
INTENT_MODEL_PATH = './models/nli-MiniLM2-L6-H768' # <<<--- UPDATE THIS PATH
EVALUATION_JSON_PATH = "./evaluation_data.json"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

LABELS_LEVEL_ONE = ["Conversation Oriented", "Task Oriented", "Entertainment"]
LABELS_LEVEL_TWO = {
    "Conversation Oriented": ["Greetings", "Farewell", "Gratitude", "Assistance", "Well-being", "Self-assessment", "Emotional-support", "Other"],
    "Task Oriented": ["System Control", "Reminder", "Search", "Information", "Navigation", "Communication", "Other"],
    "Entertainment": ["Music", "Movie", "Games", "Other"]
}
LABELS_LEVEL_THREE = {
    "Greetings": ["Formal Greeting", "Informal Greeting", "Small-talk Starter", "Other"],
    "Farewell": ["Polite Goodbye", "Casual Goodbye", "Sign-off", "Other"],
    "Gratitude": ["Expressing Thanks", "Acknowledging Help", "Other"],
    "Assistance": ["Requesting Help", "Offering Help", "Clarification Request", "Other"],
    "Well-being": ["Inquiring Health", "Expressing Concern", "Sharing Status", "Other"],
    "Self-assessment": ["Stating Capability", "Stating Limitation", "Requesting Feedback", "Other"],
    "Emotional-support": ["Offering Comfort", "Expressing Empathy", "Sharing Feelings", "Other"],
    "System Control": ["Device On", "Device Off", "Adjust Setting", "Query Status", "Other"],
    "Reminder": ["Set Reminder", "Query Reminder", "Cancel Reminder", "Other"],
    "Search": ["Web Search", "Fact Search", "Definition Search", "Other"],
    "Information": ["Requesting News", "Requesting Weather", "Requesting Time", "Requesting Facts", "Other"],
    "Navigation": ["Get Directions", "Traffic Info", "Nearby Places", "Other"],
    "Communication": ["Send Message", "Make Call", "Read Message", "Other"],
    "Music": ["Play Song", "Play Artist", "Play Genre", "Control Playback", "Other"],
    "Movie": ["Find Movie", "Movie Info", "Play Trailer", "Other"],
    "Games": ["Start Game", "Game Suggestion", "Game Score", "Other"],
    "Other": ["General Chit-Chat", "Unclassified"]
}
# Add "Other" entries if L2 categories can map to L3 "Other"
for l2_cat in LABELS_LEVEL_TWO:
    if l2_cat not in LABELS_LEVEL_THREE:
         LABELS_LEVEL_THREE[l2_cat] = ["Other", "Unclassified"]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IntentEvaluation")

# -----------------------------
# Enhanced Analysis Function
# -----------------------------
def analyze_and_explain_metrics(y_true: list, y_pred: list, level_name: str, all_labels: list | None = None):
    """
    Calculates metrics and provides a natural language explanation of the results.
    """
    logger.info(f"\n{'='*20} Detailed Analysis: {level_name} {'='*20}")

    # --- Data Filtering (same as before) ---
    filtered_true = []
    filtered_pred = []
    valid_indices = [i for i, true_label in enumerate(y_true) if true_label is not None]

    if not valid_indices:
        explanation = f"No evaluation data provided expectations for {level_name}, so no metrics could be calculated."
        logger.warning(explanation)
        print(explanation)
        return

    for i in valid_indices:
        filtered_true.append(y_true[i])
        # Use placeholder for None predictions for easier reporting
        filtered_pred.append(y_pred[i] if y_pred[i] is not None else "None_Predicted")

    present_labels = sorted(list(set(filtered_true) | set(filtered_pred)))
    num_test_cases = len(filtered_true)

    if num_test_cases == 0:
        explanation = f"After filtering, no valid data points remained for {level_name}. Cannot evaluate."
        logger.warning(explanation)
        print(explanation)
        return

    logger.info(f"Evaluating {num_test_cases} test cases for {level_name}.")

    # --- Calculate Metrics ---
    accuracy = accuracy_score(filtered_true, filtered_pred)
    report_dict = classification_report(
        filtered_true,
        filtered_pred,
        output_dict=True,
        zero_division=0,
        labels=present_labels # Important to include all relevant labels
    )

    # --- Build Natural Language Explanation ---
    explanation_lines = []

    # 1. Overall Accuracy
    accuracy_desc = "Excellent" if accuracy >= 0.9 else \
                    "Good" if accuracy >= 0.75 else \
                    "Moderate" if accuracy >= 0.6 else \
                    "Poor" if accuracy >= 0.4 else \
                    "Very Poor"
    explanation_lines.append(f"**Overall Accuracy:** {accuracy:.2%} ({accuracy_desc}). "
                             f"This means the model predicted the correct {level_name} intent "
                             f"{accuracy*100:.0f}% of the time for the evaluated cases.")

    # 2. Overall Averages (Weighted)
    w_precision = report_dict['weighted avg']['precision']
    w_recall = report_dict['weighted avg']['recall']
    w_f1 = report_dict['weighted avg']['f1-score']
    explanation_lines.append(f"\n**Overall Performance (Weighted Averages):**")
    explanation_lines.append(f"- **Weighted F1-Score:** {w_f1:.3f}. This balances precision and recall across all classes, accounting for how many examples each class had. An F1 score closer to 1 is better.")
    explanation_lines.append(f"- **Weighted Precision:** {w_precision:.3f}. On average (weighted by class size), when the model predicted an intent at this level, it was correct {w_precision*100:.0f}% of the time.")
    explanation_lines.append(f"- **Weighted Recall:** {w_recall:.3f}. On average (weighted by class size), the model correctly identified {w_recall*100:.0f}% of the actual intents present in the test data for this level.")

    # 3. Class-Specific Performance Highlights & Lowlights
    explanation_lines.append(f"\n**Performance by Specific Intent ({level_name}):**")
    found_issues = False
    good_performers = []
    poor_performers = []
    precision_recall_issues = []

    # Sort labels by support descending for reporting priority
    labels_in_report = [lbl for lbl in present_labels if lbl in report_dict]
    labels_in_report.sort(key=lambda lbl: report_dict[lbl]['support'] if isinstance(report_dict[lbl], dict) else 0, reverse=True)


    for label in labels_in_report:
        if label in ['accuracy', 'macro avg', 'weighted avg']: continue # Skip summary rows
        if not isinstance(report_dict[label], dict): continue # Ensure it's a class entry

        stats = report_dict[label]
        precision = stats['precision']
        recall = stats['recall']
        f1 = stats['f1-score']
        support = stats['support']

        if support == 0: # Skip labels not actually present in y_true (but maybe in y_pred)
             continue

        label_performance_summary = f"- **'{label}'** (Seen {support} times): F1={f1:.2f}, Precision={precision:.2f}, Recall={recall:.2f}"

        if f1 >= 0.75:
            good_performers.append(f"  - Performing well: '{label}' (F1: {f1:.2f}, Support: {support})")
        elif f1 < 0.5:
            poor_performers.append(f"  - Struggling significantly: '{label}' (F1: {f1:.2f}, Support: {support})")
            found_issues = True

        # Analyze Precision/Recall trade-offs for problematic classes
        if f1 < 0.7: # Focus analysis on less-than-good performers
            if precision < 0.5 and recall >= 0.6:
                 precision_recall_issues.append(f"  - For '{label}': High Recall / Low Precision suggests the model finds most '{label}' cases but **incorrectly labels other intents AS '{label}'** too often (many false positives).")
                 found_issues = True
            elif recall < 0.5 and precision >= 0.6:
                 precision_recall_issues.append(f"  - For '{label}': Low Recall / High Precision suggests when the model *does* predict '{label}', it's usually correct, but it **misses identifying many actual '{label}' cases** (many false negatives).")
                 found_issues = True
            elif recall < 0.5 and precision < 0.5:
                 precision_recall_issues.append(f"  - For '{label}': Low Recall / Low Precision indicates the model struggles both to find '{label}' instances and makes incorrect predictions when it does.")
                 found_issues = True

    if good_performers:
        explanation_lines.append("\n  *Good Performers:*")
        explanation_lines.extend(good_performers)
    if poor_performers:
         explanation_lines.append("\n  *Intents Needing Attention:*")
         explanation_lines.extend(poor_performers)
    if precision_recall_issues:
        explanation_lines.append("\n  *Precision/Recall Issues Analysis:*")
        explanation_lines.extend(precision_recall_issues)

    if not found_issues and accuracy > 0.8:
         explanation_lines.append("\n  *Overall Diagnosis:* Performance looks generally strong across most intents at this level.")
    elif not found_issues and accuracy <= 0.8:
        explanation_lines.append("\n  *Overall Diagnosis:* No single intent stands out as extremely problematic based on F1, but overall accuracy suggests room for improvement across the board.")
    elif found_issues:
         explanation_lines.append("\n  *Overall Diagnosis:* Specific intents listed above show significant weaknesses (low F1 and/or Precision/Recall imbalances) that are dragging down overall performance.")

    # 4. Handle "None_Predicted"
    if "None_Predicted" in report_dict:
        none_stats = report_dict["None_Predicted"]
        explanation_lines.append(f"\n**Analysis of 'None_Predicted':** (Support: {none_stats['support']})")
        explanation_lines.append(f"- Precision ({none_stats['precision']:.2f}): When the model predicted 'None', how often was the *true* label also 'None' (or not expected at this level)?")
        explanation_lines.append(f"- Recall ({none_stats['recall']:.2f}): Of the times the *true* label was 'None' (or not expected), how often did the model correctly predict 'None'?")
        explanation_lines.append(f"- This helps understand if the model fails to predict an intent when it should, or incorrectly predicts 'None' when there *was* an expected intent.")


    # 5. Hierarchical Context
    if level_name == "Level 2" or level_name == "Level 3":
        explanation_lines.append(f"\n**Hierarchical Context:** Remember that performance at {level_name} is heavily influenced by the accuracy of the predictions made at the *previous* level(s). Errors cascade down.")

    # --- Print Explanation ---
    print("\n".join(explanation_lines))

    # --- Also print the raw sklearn report for reference ---
    try:
        raw_report = classification_report(
            filtered_true,
            filtered_pred,
            zero_division=0,
            labels=present_labels
        )
        logger.info(f"\nRaw Classification Report ({level_name}):\n{raw_report}")
    except Exception as e:
        logger.error(f"Could not generate raw classification report for {level_name}: {e}")


# -------------------
# Helper Functions (Load Data, Predict - Same as before)
# -------------------
def load_test_data(json_path: str) -> list[dict] | None:
    """Loads evaluation data from a JSON file."""
    if not os.path.exists(json_path):
        logger.error(f"Evaluation data file not found: {json_path}")
        return None
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} test cases from {json_path}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {json_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error reading file {json_path}: {e}")
        return None

def predict_intent_hierarchy(classifier, text: str) -> tuple[str | None, str | None, str | None]:
    """Performs hierarchical intent prediction."""
    pred_l1, pred_l2, pred_l3 = None, None, None
    try:
        # --- Level 1 Prediction ---
        result_l1 = classifier(text, LABELS_LEVEL_ONE)
        if result_l1 and result_l1['labels']:
            pred_l1 = result_l1['labels'][0]
        else:
            logger.warning(f"Level 1 prediction failed for text: '{text}'")
            return None, None, None

        # --- Level 2 Prediction ---
        if pred_l1 in LABELS_LEVEL_TWO:
            labels_l2 = LABELS_LEVEL_TWO[pred_l1]
            result_l2 = classifier(text, labels_l2)
            if result_l2 and result_l2['labels']:
                pred_l2 = result_l2['labels'][0]
            else:
                logger.warning(f"Level 2 prediction failed for text: '{text}' (L1 was '{pred_l1}')")
        else:
            logger.debug(f"No Level 2 labels defined for predicted L1: '{pred_l1}'")

        # --- Level 3 Prediction ---
        if pred_l2 and pred_l2 in LABELS_LEVEL_THREE:
             labels_l3 = LABELS_LEVEL_THREE[pred_l2]
             result_l3 = classifier(text, labels_l3)
             if result_l3 and result_l3['labels']:
                 pred_l3 = result_l3['labels'][0]
             else:
                 logger.warning(f"Level 3 prediction failed for text: '{text}' (L1='{pred_l1}', L2='{pred_l2}')")
        elif pred_l2:
             logger.debug(f"No Level 3 labels defined for predicted L2: '{pred_l2}'")

    except Exception as e:
        logger.error(f"Error during prediction for text '{text}': {e}", exc_info=True)
        return pred_l1, pred_l2, pred_l3 # Return potentially partial predictions

    return pred_l1, pred_l2, pred_l3


# -------------------
# Main Execution (Modified to use new analysis function)
# -------------------
if __name__ == "__main__":
    logger.info("Starting Intent Classification Evaluation...")

    # --- Load Model ---
    if not os.path.exists(INTENT_MODEL_PATH):
        logger.critical(f"Intent model path not found: {INTENT_MODEL_PATH}")
        exit(1)
    try:
        logger.info(f"Loading intent classifier model from: {INTENT_MODEL_PATH} on device: {DEVICE}")
        # Specify device ID if using CUDA and transformers > 4.0 approx
        device_arg = 0 if DEVICE.startswith("cuda") else -1 # pipeline uses -1 for CPU, 0 for cuda:0 etc.
        intent_classifier = pipeline(
            "zero-shot-classification",
            model=INTENT_MODEL_PATH,
            device=device_arg # Use device argument
        )
        logger.info(f"Intent classifier loaded successfully onto device index: {device_arg} ({DEVICE}).")
    except Exception as e:
        logger.critical(f"Failed to load intent classifier model: {e}", exc_info=True)
        exit(1)

    # --- Load Data ---
    test_data = load_test_data(EVALUATION_JSON_PATH)
    if not test_data:
        exit(1)

    # --- Run Predictions ---
    results = []
    y_true_l1, y_pred_l1 = [], []
    y_true_l2, y_pred_l2 = [], []
    y_true_l3, y_pred_l3 = [], []

    logger.info("Running predictions on test data...")
    for i, item in enumerate(test_data):
        question = item.get("question")
        if not question:
            logger.warning(f"Skipping item {i+1}: Missing 'question'.")
            continue

        expected_l1 = item.get("expected_level_1")
        expected_l2 = item.get("expected_level_2")
        expected_l3 = item.get("expected_level_3")

        if not expected_l1:
             logger.warning(f"Skipping item {i+1} ('{question}'): Missing 'expected_level_1'.")
             continue

        pred_l1, pred_l2, pred_l3 = predict_intent_hierarchy(intent_classifier, question)

        results.append({
            "question": question,
            "expected_l1": expected_l1, "predicted_l1": pred_l1,
            "expected_l2": expected_l2, "predicted_l2": pred_l2,
            "expected_l3": expected_l3, "predicted_l3": pred_l3,
            "correct_l1": expected_l1 == pred_l1,
            "correct_l2": expected_l2 == pred_l2,
            "correct_l3": expected_l3 == pred_l3,
        })

        y_true_l1.append(expected_l1)
        y_pred_l1.append(pred_l1)
        y_true_l2.append(expected_l2)
        y_pred_l2.append(pred_l2)
        y_true_l3.append(expected_l3)
        y_pred_l3.append(pred_l3)

        if (i + 1) % 20 == 0:
            logger.info(f"Processed {i+1}/{len(test_data)} items...")

    logger.info("Prediction complete.")

    # --- Calculate Metrics and Get Explanations ---
    all_l1_labels = LABELS_LEVEL_ONE
    all_l2_labels = list(set(lbl for sublist in LABELS_LEVEL_TWO.values() for lbl in sublist))
    all_l3_labels = list(set(lbl for sublist in LABELS_LEVEL_THREE.values() for lbl in sublist))

    analyze_and_explain_metrics(y_true_l1, y_pred_l1, "Level 1", all_labels=all_l1_labels)
    analyze_and_explain_metrics(y_true_l2, y_pred_l2, "Level 2", all_labels=all_l2_labels)
    analyze_and_explain_metrics(y_true_l3, y_pred_l3, "Level 3", all_labels=all_l3_labels)

    # --- Optional: Save detailed results (same as before) ---
    # with open("evaluation_results_detailed.json", "w", encoding="utf-8") as f:
    #     json.dump(results, f, indent=4)
    # logger.info("Detailed results saved to evaluation_results_detailed.json")

    logger.info("Intent Classification Evaluation Finished.")


















































































# import json
# import os
# import logging
# from transformers import pipeline
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
# import torch # Ensure torch is imported if using GPU/CPU specification

# # -------------------
# # Configuration
# # -------------------
# # --- Model Path (MAKE SURE THIS IS CORRECT) ---
# INTENT_MODEL_PATH = './models/nli-MiniLM2-L6-H768'
# # Or use a relative path if appropriate: './Res/LLMs/Encoder/nli-MiniLM2-L6-H768'

# # --- Evaluation Data File ---
# EVALUATION_JSON_PATH = "./evaluation_data.json"

# # --- Device ---
# # Use "cuda:0" if GPU is available and desired, otherwise "cpu"
# DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# # --- Label Definitions (Copy from the original script) ---
# LABELS_LEVEL_ONE = ["Conversation Oriented", "Task Oriented", "Entertainment"]
# LABELS_LEVEL_TWO = {
#     "Conversation Oriented": ["Greetings", "Farewell", "Gratitude", "Assistance", "Well-being", "Self-assessment", "Emotional-support", "Other"],
#     "Task Oriented": ["System Control", "Reminder", "Search", "Information", "Navigation", "Communication", "Other"],
#     "Entertainment": ["Music", "Movie", "Games", "Other"]
# }
# LABELS_LEVEL_THREE = {
#     "Greetings": ["Formal Greeting", "Informal Greeting", "Small-talk Starter", "Other"],
#     "Farewell": ["Polite Goodbye", "Casual Goodbye", "Sign-off", "Other"],
#     "Gratitude": ["Expressing Thanks", "Acknowledging Help", "Other"],
#     "Assistance": ["Requesting Help", "Offering Help", "Clarification Request", "Other"],
#     "Well-being": ["Inquiring Health", "Expressing Concern", "Sharing Status", "Other"],
#     "Self-assessment": ["Stating Capability", "Stating Limitation", "Requesting Feedback", "Other"],
#     "Emotional-support": ["Offering Comfort", "Expressing Empathy", "Sharing Feelings", "Other"],
#     "System Control": ["Device On", "Device Off", "Adjust Setting", "Query Status", "Other"],
#     "Reminder": ["Set Reminder", "Query Reminder", "Cancel Reminder", "Other"],
#     "Search": ["Web Search", "Fact Search", "Definition Search", "Other"],
#     "Information": ["Requesting News", "Requesting Weather", "Requesting Time", "Requesting Facts", "Other"],
#     "Navigation": ["Get Directions", "Traffic Info", "Nearby Places", "Other"],
#     "Communication": ["Send Message", "Make Call", "Read Message", "Other"],
#     "Music": ["Play Song", "Play Artist", "Play Genre", "Control Playback", "Other"],
#     "Movie": ["Find Movie", "Movie Info", "Play Trailer", "Other"],
#     "Games": ["Start Game", "Game Suggestion", "Game Score", "Other"],
#     "Other": ["General Chit-Chat", "Unclassified"] # Note: Ensure "Other" covers L2/L3 too if needed
# }
# # Add "Other" entries if L2 categories can map to L3 "Other"
# for l2_cat in LABELS_LEVEL_TWO:
#     if l2_cat not in LABELS_LEVEL_THREE:
#          LABELS_LEVEL_THREE[l2_cat] = ["Other", "Unclassified"] # Default L3 if L2 is not explicitly defined in L3 dict

# # --- Logging ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger("IntentEvaluation")

# # -------------------
# # Helper Functions
# # -------------------

# def load_test_data(json_path: str) -> list[dict] | None:
#     """Loads evaluation data from a JSON file."""
#     if not os.path.exists(json_path):
#         logger.error(f"Evaluation data file not found: {json_path}")
#         return None
#     try:
#         with open(json_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#         logger.info(f"Loaded {len(data)} test cases from {json_path}")
#         return data
#     except json.JSONDecodeError as e:
#         logger.error(f"Error decoding JSON from {json_path}: {e}")
#         return None
#     except Exception as e:
#         logger.error(f"Error reading file {json_path}: {e}")
#         return None

# def predict_intent_hierarchy(classifier, text: str) -> tuple[str | None, str | None, str | None]:
#     """Performs hierarchical intent prediction."""
#     pred_l1, pred_l2, pred_l3 = None, None, None
#     try:
#         # --- Level 1 Prediction ---
#         result_l1 = classifier(text, LABELS_LEVEL_ONE)
#         if result_l1 and result_l1['labels']:
#             pred_l1 = result_l1['labels'][0]
#         else:
#             logger.warning(f"Level 1 prediction failed for text: '{text}'")
#             return None, None, None # Cannot proceed

#         # --- Level 2 Prediction ---
#         if pred_l1 in LABELS_LEVEL_TWO:
#             labels_l2 = LABELS_LEVEL_TWO[pred_l1]
#             result_l2 = classifier(text, labels_l2)
#             if result_l2 and result_l2['labels']:
#                 pred_l2 = result_l2['labels'][0]
#             else:
#                 logger.warning(f"Level 2 prediction failed for text: '{text}' (L1 was '{pred_l1}')")
#         else:
#             logger.debug(f"No Level 2 labels defined for predicted L1: '{pred_l1}'")

#         # --- Level 3 Prediction ---
#         if pred_l2 and pred_l2 in LABELS_LEVEL_THREE:
#              labels_l3 = LABELS_LEVEL_THREE[pred_l2]
#              result_l3 = classifier(text, labels_l3)
#              if result_l3 and result_l3['labels']:
#                  pred_l3 = result_l3['labels'][0]
#              else:
#                  logger.warning(f"Level 3 prediction failed for text: '{text}' (L1='{pred_l1}', L2='{pred_l2}')")
#         elif pred_l2:
#              logger.debug(f"No Level 3 labels defined for predicted L2: '{pred_l2}'")

#     except Exception as e:
#         logger.error(f"Error during prediction for text '{text}': {e}", exc_info=True)
#         # Return potentially partial predictions made before the error
#         return pred_l1, pred_l2, pred_l3

#     return pred_l1, pred_l2, pred_l3

# def calculate_and_print_metrics(y_true: list, y_pred: list, level_name: str, all_labels: list | None = None):
#     """Calculates and prints evaluation metrics for a given level."""
#     logger.info(f"\n--- Evaluating {level_name} ---")

#     # Filter out entries where expected label is None
#     # We evaluate only when an expectation exists for this level.
#     # We keep the prediction even if it's None, to penalize wrong classifications
#     # where the model predicted something but shouldn't have, or vice-versa.
#     # However, sklearn metrics often work best with string labels, so handle None predictions.
#     filtered_true = []
#     filtered_pred = []
#     valid_indices = [i for i, true_label in enumerate(y_true) if true_label is not None]

#     if not valid_indices:
#         logger.warning(f"No valid test cases with expected labels found for {level_name}. Skipping metrics.")
#         return

#     for i in valid_indices:
#         filtered_true.append(y_true[i])
#         # Replace None predictions with a placeholder string if needed by sklearn,
#         # or handle them based on how you want to evaluate (e.g., consider it incorrect)
#         # Using a placeholder string is often easiest for classification_report.
#         filtered_pred.append(y_pred[i] if y_pred[i] is not None else "None_Predicted")

#     # Get the unique set of labels present in the filtered true/pred lists + placeholder
#     present_labels = sorted(list(set(filtered_true) | set(filtered_pred)))

#     if not filtered_true: # Double check after potential None filtering in pred
#         logger.warning(f"No valid data points remain for {level_name} after filtering. Skipping metrics.")
#         return

#     # --- Calculate Metrics ---
#     accuracy = accuracy_score(filtered_true, filtered_pred)
#     precision, recall, f1, _ = precision_recall_fscore_support(
#         filtered_true,
#         filtered_pred,
#         average='weighted', # Use 'weighted' for multiclass, considers class imbalance
#         zero_division=0,    # Avoids warning if a class has no predicted samples
#         labels=present_labels # Ensure all relevant labels are considered
#     )

#     logger.info(f"Accuracy: {accuracy:.4f}")
#     logger.info(f"Weighted Precision: {precision:.4f}")
#     logger.info(f"Weighted Recall: {recall:.4f}")
#     logger.info(f"Weighted F1-Score: {f1:.4f}")

#     # --- Detailed Report (Per Class) ---
#     try:
#         report = classification_report(
#             filtered_true,
#             filtered_pred,
#             zero_division=0,
#             labels=present_labels # Explicitly pass labels
#         )
#         logger.info(f"Classification Report:\n{report}")
#     except Exception as e:
#         logger.error(f"Could not generate classification report for {level_name}: {e}")

# # -------------------
# # Main Execution
# # -------------------
# if __name__ == "__main__":
#     logger.info("Starting Intent Classification Evaluation...")

#     # --- Load Model ---
#     if not os.path.exists(INTENT_MODEL_PATH):
#         logger.critical(f"Intent model path not found: {INTENT_MODEL_PATH}")
#         exit(1)
#     try:
#         logger.info(f"Loading intent classifier model from: {INTENT_MODEL_PATH} on device: {DEVICE}")
#         intent_classifier = pipeline(
#             "zero-shot-classification",
#             model=INTENT_MODEL_PATH,
#             device=DEVICE # Use 0 for cuda:0, 1 for cuda:1 etc., or -1 for CPU if not using torch import method
#         )
#         logger.info("Intent classifier loaded successfully.")
#     except Exception as e:
#         logger.critical(f"Failed to load intent classifier model: {e}", exc_info=True)
#         exit(1)

#     # --- Load Data ---
#     test_data = load_test_data(EVALUATION_JSON_PATH)
#     if not test_data:
#         exit(1)

#     # --- Run Predictions ---
#     results = []
#     y_true_l1, y_pred_l1 = [], []
#     y_true_l2, y_pred_l2 = [], []
#     y_true_l3, y_pred_l3 = [], []

#     logger.info("Running predictions on test data...")
#     for i, item in enumerate(test_data):
#         question = item.get("question")
#         if not question:
#             logger.warning(f"Skipping item {i+1}: Missing 'question'.")
#             continue

#         expected_l1 = item.get("expected_level_1") # Must exist
#         expected_l2 = item.get("expected_level_2") # Can be None
#         expected_l3 = item.get("expected_level_3") # Can be None

#         if not expected_l1:
#              logger.warning(f"Skipping item {i+1} ('{question}'): Missing 'expected_level_1'.")
#              continue

#         pred_l1, pred_l2, pred_l3 = predict_intent_hierarchy(intent_classifier, question)

#         results.append({
#             "question": question,
#             "expected_l1": expected_l1, "predicted_l1": pred_l1,
#             "expected_l2": expected_l2, "predicted_l2": pred_l2,
#             "expected_l3": expected_l3, "predicted_l3": pred_l3,
#             "correct_l1": expected_l1 == pred_l1,
#             "correct_l2": expected_l2 == pred_l2,
#             "correct_l3": expected_l3 == pred_l3,
#         })

#         # Store for sklearn metrics
#         y_true_l1.append(expected_l1)
#         y_pred_l1.append(pred_l1)
#         y_true_l2.append(expected_l2)
#         y_pred_l2.append(pred_l2)
#         y_true_l3.append(expected_l3)
#         y_pred_l3.append(pred_l3)

#         # Log progress occasionally
#         if (i + 1) % 20 == 0:
#             logger.info(f"Processed {i+1}/{len(test_data)} items...")

#     logger.info("Prediction complete.")

#     # --- Calculate and Print Metrics ---
#     # Define all possible labels that *could* appear at each level for reporting completeness
#     all_l1_labels = LABELS_LEVEL_ONE
#     all_l2_labels = list(set(lbl for sublist in LABELS_LEVEL_TWO.values() for lbl in sublist))
#     all_l3_labels = list(set(lbl for sublist in LABELS_LEVEL_THREE.values() for lbl in sublist))

#     calculate_and_print_metrics(y_true_l1, y_pred_l1, "Level 1", all_labels=all_l1_labels)
#     calculate_and_print_metrics(y_true_l2, y_pred_l2, "Level 2", all_labels=all_l2_labels)
#     calculate_and_print_metrics(y_true_l3, y_pred_l3, "Level 3", all_labels=all_l3_labels)

#     # --- Optional: Save detailed results ---
#     # You could save the 'results' list to a JSON file for detailed inspection
#     # with open("evaluation_results_detailed.json", "w", encoding="utf-8") as f:
#     #     json.dump(results, f, indent=4)
#     # logger.info("Detailed results saved to evaluation_results_detailed.json")

#     logger.info("Intent Classification Evaluation Finished.")