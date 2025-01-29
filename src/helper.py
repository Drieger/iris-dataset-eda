from IPython.display import HTML
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score

def display_inline(dataframes, titles):
    """Display dataframes side by side"""
    html = '<div style="display:flex">'
    for df, title in zip(dataframes, titles):
        html += '<div style="margin-right: 2em; text-align=center">'
        html += '<div style="text-align:center">' + title + '</div>'
        html += df.to_html()
        html += '</div>'
    html += '</div>'
    return display(HTML(html))

def print_metrics(y, y_pred, title='Metrics', average='micro'):
    """Print relevant metrics"""
    print(title)
    print("=" * len(title))
    print("Accuracy: {:.2%}".format(accuracy_score(y, y_pred)))
    print("Balanced accuracy: {:.2%}".format(balanced_accuracy_score(y, y_pred)))
    print("Precision: {:.2%}".format(precision_score(y, y_pred, average=average)))
    print("Recall: {:.2%}".format(recall_score(y, y_pred, average=average)))
    print("F1 Score: {:.2%}".format(f1_score(y, y_pred, average=average)))
    print("-" * len(title))