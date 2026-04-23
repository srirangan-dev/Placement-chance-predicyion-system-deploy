import os
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt


import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import warnings

warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             confusion_matrix)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from matplotlib.patches import Patch
from itertools import combinations


# ─── Load Data ─────────────────────────────────────────────
base_path = os.path.dirname(os.path.abspath(__file__))

train = pd.read_csv(os.path.join(base_path, "train.csv"))
test  = pd.read_csv(os.path.join(base_path, "test.csv"))

print(f"Train: {train.shape}  |  Test: {test.shape}")
print(f"Columns: {list(train.columns)}")

print("\nPlacement distribution:")
print(train['Placement_Status'].value_counts())


# ─── Preprocessing ─────────────────────────────────────────
drop_cols = ['Student_ID']
train.drop(columns=drop_cols, inplace=True, errors='ignore')
test.drop(columns=drop_cols, inplace=True, errors='ignore')

cat_cols = ['Gender', 'Degree', 'Branch']
encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col]  = le.transform(test[col])
    encoders[col] = le

target_enc = LabelEncoder()
train['Placement_Status'] = target_enc.fit_transform(train['Placement_Status'])

print("\nLabel encoding:")
print(dict(zip(target_enc.classes_, target_enc.transform(target_enc.classes_))))


# ─── Feature & Target ─────────────────────────────────────
feature_cols = [c for c in train.columns if c != 'Placement_Status']

X_train = train[feature_cols]
y_train = train['Placement_Status']

if 'Placement_Status' in test.columns:
    test['Placement_Status'] = target_enc.transform(test['Placement_Status'])
    X_test = test[feature_cols]
    y_test = test['Placement_Status']
else:
    X_test = test[feature_cols]
    y_test = None


# ─── Leakage Check ─────────────────────────────────────────
print("\n🔍 Correlation with Target:")
print(train.corr()['Placement_Status'].sort_values(ascending=False))


# ─── Models ───────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
}

results = []
trained_models = {}

print("\n" + "="*60)
print("Training Models...")
print("="*60)


# ─── Training Loop ─────────────────────────────────────────
for name, model in models.items():

    model.fit(X_train, y_train)
    trained_models[name] = model

    train_acc = accuracy_score(y_train, model.predict(X_train))

    if y_test is not None:
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        acc   = accuracy_score(y_test, preds)
        f1    = f1_score(y_test, preds, average='weighted')
        auc   = roc_auc_score(y_test, proba)
    else:
        preds = model.predict(X_train)
        proba = model.predict_proba(X_train)[:, 1]
        acc   = train_acc
        f1    = f1_score(y_train, preds, average='weighted')
        auc   = roc_auc_score(y_train, proba)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

    results.append({
        'Model':          name,
        'Accuracy':       acc,
        'Train_Accuracy': train_acc,
        'F1':             f1,
        'AUC':            auc,
        'CV_Mean':        cv_scores.mean(),
        'CV_Std':         cv_scores.std()
    })

    print(f"{name:<22} Train: {train_acc:.4f} | Test: {acc:.4f} | CV: {cv_scores.mean():.4f}")


# ─── Results DataFrame ─────────────────────────────────────
results_df = pd.DataFrame(results).sort_values(
    by=['Accuracy', 'CV_Mean', 'AUC'], ascending=False
)

print("\n📊 Model Comparison:")
print(results_df.to_string(index=False))


# ─── Best Model Selection ──────────────────────────────────
top_acc    = results_df.iloc[0]['Accuracy']
top_models = results_df[results_df['Accuracy'] == top_acc]
best_name  = 'Random Forest' if 'Random Forest' in top_models['Model'].values else top_models.iloc[0]['Model']
best_model = trained_models[best_name]

print(f"\n🏆 Best Model Selected: {best_name}")


# ─── Feature Importance ────────────────────────────────────
fi = None
if hasattr(best_model, 'feature_importances_'):
    fi = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\n📌 Feature Importances:")
    for feat, imp in fi.items():
        bar = '█' * int(imp * 100)
        print(f"  {feat:<25} {imp:.4f}  {bar}")

    joblib.dump(fi, os.path.join(base_path, 'feature_importances.pkl'))
    print("\n✅ Feature importances saved → feature_importances.pkl")


# ─── Save Model Files ──────────────────────────────────────
joblib.dump(best_model,   os.path.join(base_path, 'best_model.pkl'))
joblib.dump(encoders,     os.path.join(base_path, 'label_encoders.pkl'))
joblib.dump(target_enc,   os.path.join(base_path, 'target_encoder.pkl'))
joblib.dump(feature_cols, os.path.join(base_path, 'feature_cols.pkl'))
print("✅ Model files saved")


# ══════════════════════════════════════════════════════════════
#  COLOUR PALETTE & HELPERS
# ══════════════════════════════════════════════════════════════
COLORS   = ['#4fc3f7','#81c784','#ffb74d','#e57373','#ce93d8','#80cbc4']
TEXT_C   = '#e6edf3'
GRID_C   = '#30363d'
BG       = '#161b22'
PAGE_BG  = '#0d1117'
PLACED   = '#4fc3f7'   # placed students  → cyan
NOTPLACE = '#f87171'   # not placed        → red

# Class labels & numeric targets for scatter colouring
class_labels = target_enc.classes_          # e.g. ['Not Placed', 'Placed']
y_col        = train['Placement_Status']    # 0/1

SCATTER_PALETTE = {0: NOTPLACE, 1: PLACED}
scatter_colors  = y_col.map(SCATTER_PALETTE)

def style_ax(ax, title):
    ax.set_facecolor(BG)
    ax.set_title(title, color=TEXT_C, fontsize=10, fontweight='bold', pad=10)
    ax.tick_params(colors=TEXT_C, labelsize=8)
    ax.grid(True, color=GRID_C, linewidth=0.5, alpha=0.6)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_C)
    ax.xaxis.label.set_color(TEXT_C)
    ax.yaxis.label.set_color(TEXT_C)

def scatter_legend(ax):
    handles = [
        plt.Line2D([0],[0], marker='o', color='w',
                   markerfacecolor=PLACED,   markersize=7, label=class_labels[1]),
        plt.Line2D([0],[0], marker='o', color='w',
                   markerfacecolor=NOTPLACE, markersize=7, label=class_labels[0]),
    ]
    ax.legend(handles=handles, loc='best', framealpha=0.2,
              labelcolor=TEXT_C, fontsize=7.5)


# ══════════════════════════════════════════════════════════════
#  REPORT 1  –  Original 5-panel report  (ml_report.png)
# ══════════════════════════════════════════════════════════════
model_names = results_df['Model'].tolist()

fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor(PAGE_BG)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.38)

# ── 1. Accuracy ────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(model_names, results_df['Accuracy']*100,
               color=COLORS[:len(model_names)], width=0.55, edgecolor='none')
ax1.set_ylim(max(0, results_df['Accuracy'].min()*100-5), 101)
ax1.set_ylabel('Accuracy %', color=TEXT_C, fontsize=8)
for bar, val in zip(bars, results_df['Accuracy']):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
             f'{val*100:.1f}%', ha='center', va='bottom', color=TEXT_C, fontsize=7.5)
ax1.set_xticklabels(model_names, rotation=30, ha='right')
style_ax(ax1, '📊 Test Accuracy by Model')

# ── 2. Cross-Validation ────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
bars2 = ax2.bar(model_names, results_df['CV_Mean']*100,
                color=COLORS[:len(model_names)], width=0.55, edgecolor='none')
ax2.errorbar(model_names, results_df['CV_Mean']*100,
             yerr=results_df['CV_Std']*100,
             fmt='none', color='white', capsize=4, linewidth=1.2)
ax2.set_ylim(max(0,(results_df['CV_Mean']-results_df['CV_Std']).min()*100-5), 101)
ax2.set_ylabel('CV Accuracy %', color=TEXT_C, fontsize=8)
for bar, val in zip(bars2, results_df['CV_Mean']):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
             f'{val*100:.1f}%', ha='center', va='bottom', color=TEXT_C, fontsize=7.5)
ax2.set_xticklabels(model_names, rotation=30, ha='right')
style_ax(ax2, '🔁 Cross-Validation Score (±std)')

# ── 3. AUC-ROC ─────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
bars3 = ax3.bar(model_names, results_df['AUC']*100,
                color=COLORS[:len(model_names)], width=0.55, edgecolor='none')
ax3.set_ylim(max(0, results_df['AUC'].min()*100-5), 101)
ax3.set_ylabel('AUC-ROC %', color=TEXT_C, fontsize=8)
for bar, val in zip(bars3, results_df['AUC']):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
             f'{val*100:.1f}%', ha='center', va='bottom', color=TEXT_C, fontsize=7.5)
ax3.set_xticklabels(model_names, rotation=30, ha='right')
style_ax(ax3, '🎯 AUC-ROC Score')

# ── 4. Feature Importance ─────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0:2])
if fi is not None:
    fi_sorted  = fi.sort_values(ascending=True)
    bar_colors = ['#f87171' if fi_sorted[f]==fi_sorted.max() else
                  '#4fc3f7' if fi_sorted[f]>=fi_sorted.quantile(0.75) else
                  '#6366f1' for f in fi_sorted.index]
    h_bars = ax4.barh(fi_sorted.index, fi_sorted.values*100,
                      color=bar_colors, height=0.55, edgecolor='none')
    ax4.set_xlabel('Importance (%)', color=TEXT_C, fontsize=9)
    for bar, val in zip(h_bars, fi_sorted.values):
        ax4.text(val*100+0.2, bar.get_y()+bar.get_height()/2,
                 f'{val*100:.2f}%', va='center', color=TEXT_C, fontsize=8.5)
    ax4.set_xlim(0, fi_sorted.max()*100*1.22)
    legend_els = [Patch(color='#f87171', label='Most Important'),
                  Patch(color='#4fc3f7', label='High Importance'),
                  Patch(color='#6366f1', label='Moderate Importance')]
    ax4.legend(handles=legend_els, loc='lower right',
               framealpha=0.15, labelcolor=TEXT_C, fontsize=8)
style_ax(ax4, f'🌟 Feature Importance — Which Fields Matter Most? ({best_name})')

# ── 5. Confusion Matrix ───────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
eval_X = X_test if y_test is not None else X_train
eval_y = y_test if y_test is not None else y_train
cm = confusion_matrix(eval_y, best_model.predict(eval_X))
labels = target_enc.classes_
sns.heatmap(cm, annot=True, fmt='d', ax=ax5,
            cmap='Blues', linewidths=0.5,
            xticklabels=labels, yticklabels=labels,
            annot_kws={'size':12,'weight':'bold'})
ax5.set_xlabel('Predicted', color=TEXT_C, fontsize=9)
ax5.set_ylabel('Actual',    color=TEXT_C, fontsize=9)
ax5.tick_params(colors=TEXT_C)
ax5.set_facecolor(BG)
ax5.set_title('🧩 Confusion Matrix', color=TEXT_C, fontsize=11,
              fontweight='bold', pad=10)

fig.suptitle('Placement Prediction — ML Model Report',
             color=TEXT_C, fontsize=16, fontweight='bold', y=1.01)

plt.savefig(os.path.join(base_path, 'ml_report.png'),
            dpi=150, bbox_inches='tight', facecolor=PAGE_BG)
plt.close()
print("✅ ml_report.png saved")


# ══════════════════════════════════════════════════════════════
#  REPORT 2  –  Scatter Plot Report  (scatter_report.png)
#
#  Layout  (4 rows × 3 cols  =  12 panels)
#  Row 0 : Top-3 most-important feature pairs (by fi rank)
#  Row 1 : 3 more numeric-feature pair scatters
#  Row 2 : PCA 2-D scatter  |  CGPA vs Placement (box/strip)  |  Histograms overlay
#  Row 3 : Model Metrics scatter  |  Train vs Test scatter  |  Prediction confidence scatter
# ══════════════════════════════════════════════════════════════

# ── identify numeric-only feature columns ──────────────────
numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()

# pick top feature pairs by importance if available, else just sequential
if fi is not None:
    top_feats = fi.sort_values(ascending=False).index.tolist()
else:
    top_feats = numeric_features

# generate ordered unique pairs from top features
ordered_pairs = [(a, b) for a, b in combinations(top_feats, 2)
                 if a in numeric_features and b in numeric_features]

# ── fig setup ─────────────────────────────────────────────
fig2 = plt.figure(figsize=(22, 28))
fig2.patch.set_facecolor(PAGE_BG)
gs2  = gridspec.GridSpec(4, 3, figure=fig2, hspace=0.55, wspace=0.38)

# ─────────────────────────────────────────────────────────
# ROW 0  — Top 3 important feature-pair scatter plots
# ─────────────────────────────────────────────────────────
for col_idx in range(3):
    ax = fig2.add_subplot(gs2[0, col_idx])
    if col_idx < len(ordered_pairs):
        fx, fy = ordered_pairs[col_idx]
        for cls in [0, 1]:
            mask = y_col == cls
            ax.scatter(X_train.loc[mask, fx], X_train.loc[mask, fy],
                       c=SCATTER_PALETTE[cls], alpha=0.55, s=22,
                       edgecolors='none', label=class_labels[cls])
        ax.set_xlabel(fx, fontsize=8)
        ax.set_ylabel(fy, fontsize=8)
        scatter_legend(ax)
        style_ax(ax, f'🔵 {fx}  vs  {fy}')
    else:
        ax.set_visible(False)

# ─────────────────────────────────────────────────────────
# ROW 1  — Next 3 feature pairs
# ─────────────────────────────────────────────────────────
for col_idx in range(3):
    ax = fig2.add_subplot(gs2[1, col_idx])
    pair_idx = 3 + col_idx
    if pair_idx < len(ordered_pairs):
        fx, fy = ordered_pairs[pair_idx]
        for cls in [0, 1]:
            mask = y_col == cls
            ax.scatter(X_train.loc[mask, fx], X_train.loc[mask, fy],
                       c=SCATTER_PALETTE[cls], alpha=0.55, s=22,
                       edgecolors='none', label=class_labels[cls])
        ax.set_xlabel(fx, fontsize=8)
        ax.set_ylabel(fy, fontsize=8)
        scatter_legend(ax)
        style_ax(ax, f'🔵 {fx}  vs  {fy}')
    else:
        ax.set_visible(False)

# ─────────────────────────────────────────────────────────
# ROW 2, Col 0  — PCA 2-D projection scatter
# ─────────────────────────────────────────────────────────
ax_pca = fig2.add_subplot(gs2[2, 0])
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_train[numeric_features])
var_explained = pca.explained_variance_ratio_ * 100

for cls in [0, 1]:
    mask = (y_col == cls).values
    ax_pca.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=SCATTER_PALETTE[cls], alpha=0.55, s=22,
                   edgecolors='none', label=class_labels[cls])
ax_pca.set_xlabel(f'PC1 ({var_explained[0]:.1f}% var)', fontsize=8)
ax_pca.set_ylabel(f'PC2 ({var_explained[1]:.1f}% var)', fontsize=8)
scatter_legend(ax_pca)
style_ax(ax_pca, '🧭 PCA 2-D Projection (All Features)')

# ─────────────────────────────────────────────────────────
# ROW 2, Col 1  — Feature vs Placement probability (jitter strip)
# ─────────────────────────────────────────────────────────
ax_strip = fig2.add_subplot(gs2[2, 1])
# use the most important numeric feature
top_num_feat = next((f for f in top_feats if f in numeric_features), numeric_features[0])
jitter = np.random.uniform(-0.12, 0.12, size=len(y_col))
for cls in [0, 1]:
    mask = y_col == cls
    ax_strip.scatter(X_train.loc[mask, top_num_feat],
                     y_col[mask] + jitter[mask.values],
                     c=SCATTER_PALETTE[cls], alpha=0.45, s=18,
                     edgecolors='none', label=class_labels[cls])
ax_strip.set_yticks([0, 1])
ax_strip.set_yticklabels(class_labels, color=TEXT_C, fontsize=8)
ax_strip.set_xlabel(top_num_feat, fontsize=8)
ax_strip.set_ylabel('Placement Status', fontsize=8)
scatter_legend(ax_strip)
style_ax(ax_strip, f'📍 {top_num_feat} vs Placement (Jitter Strip)')

# ─────────────────────────────────────────────────────────
# ROW 2, Col 2  — Prediction Confidence scatter (best model)
# ─────────────────────────────────────────────────────────
ax_conf = fig2.add_subplot(gs2[2, 2])
proba_train = best_model.predict_proba(X_train)[:, 1]
x_conf = np.arange(len(proba_train))

# colour by correct / incorrect prediction
preds_train = best_model.predict(X_train)
correct_mask = (preds_train == y_train.values)

ax_conf.scatter(x_conf[correct_mask],  proba_train[correct_mask],
                c='#4fc3f7', alpha=0.4, s=10, label='Correct')
ax_conf.scatter(x_conf[~correct_mask], proba_train[~correct_mask],
                c='#f87171', alpha=0.6, s=14, marker='x', label='Incorrect')
ax_conf.axhline(0.5, color='#ffb74d', linewidth=1.2, linestyle='--', alpha=0.7)
ax_conf.set_xlabel('Sample Index', fontsize=8)
ax_conf.set_ylabel('Predicted Prob (Placed)', fontsize=8)
ax_conf.legend(loc='best', framealpha=0.2, labelcolor=TEXT_C, fontsize=7.5)
style_ax(ax_conf, f'🎯 Confidence Scatter — {best_name}')

# ─────────────────────────────────────────────────────────
# ROW 3, Col 0  — Model Metrics bubble scatter
# ─────────────────────────────────────────────────────────
ax_metrics = fig2.add_subplot(gs2[3, 0])
metric_x = results_df['CV_Mean'].values * 100
metric_y = results_df['AUC'].values    * 100
metric_s = results_df['F1'].values     * 600    # bubble size

sc = ax_metrics.scatter(metric_x, metric_y, s=metric_s,
                        c=COLORS[:len(results_df)], alpha=0.80, edgecolors='white',
                        linewidths=0.5)
for i, row in results_df.iterrows():
    ax_metrics.annotate(row['Model'],
                        (row['CV_Mean']*100, row['AUC']*100),
                        textcoords='offset points', xytext=(6, 4),
                        color=TEXT_C, fontsize=7)
ax_metrics.set_xlabel('CV Accuracy %', fontsize=8)
ax_metrics.set_ylabel('AUC-ROC %',    fontsize=8)
style_ax(ax_metrics, '🫧 Model Metrics Bubble (size = F1)')

# ─────────────────────────────────────────────────────────
# ROW 3, Col 1  — Train vs Test Accuracy scatter
# ─────────────────────────────────────────────────────────
ax_tt = fig2.add_subplot(gs2[3, 1])
ax_tt.scatter(results_df['Train_Accuracy']*100, results_df['Accuracy']*100,
              c=COLORS[:len(results_df)], s=90, edgecolors='white',
              linewidths=0.6, alpha=0.9, zorder=3)

# ideal line  y = x
lims = [min(results_df[['Train_Accuracy','Accuracy']].min())*100 - 2,
        max(results_df[['Train_Accuracy','Accuracy']].max())*100 + 2]
ax_tt.plot(lims, lims, '--', color='#ffb74d', linewidth=1.2,
           alpha=0.7, label='Train = Test')
ax_tt.set_xlim(lims); ax_tt.set_ylim(lims)
ax_tt.set_xlabel('Train Accuracy %', fontsize=8)
ax_tt.set_ylabel('Test Accuracy %',  fontsize=8)
for i, row in results_df.iterrows():
    ax_tt.annotate(row['Model'],
                   (row['Train_Accuracy']*100, row['Accuracy']*100),
                   textcoords='offset points', xytext=(5, 3),
                   color=TEXT_C, fontsize=6.5)
ax_tt.legend(loc='lower right', framealpha=0.2, labelcolor=TEXT_C, fontsize=7.5)
style_ax(ax_tt, '🏋️ Train vs Test Accuracy (overfit check)')

# ─────────────────────────────────────────────────────────
# ROW 3, Col 2  — Feature Correlation heat-scatter (pairwise corr)
# ─────────────────────────────────────────────────────────
ax_corr = fig2.add_subplot(gs2[3, 2])
corr_mat = train[numeric_features + ['Placement_Status']].corr()
corr_vals = corr_mat['Placement_Status'].drop('Placement_Status').sort_values()
colors_corr = ['#f87171' if v < 0 else '#4fc3f7' for v in corr_vals.values]
ax_corr.barh(corr_vals.index, corr_vals.values,
             color=colors_corr, height=0.55, edgecolor='none')
ax_corr.axvline(0, color='#ffb74d', linewidth=1.2, linestyle='--', alpha=0.8)
ax_corr.set_xlabel('Pearson Correlation with Placement', fontsize=8)
for i, (feat, val) in enumerate(corr_vals.items()):
    ax_corr.text(val + (0.005 if val >= 0 else -0.005),
                 i, f'{val:.3f}',
                 va='center', ha='left' if val >= 0 else 'right',
                 color=TEXT_C, fontsize=7.5)
style_ax(ax_corr, '📐 Feature Correlation with Placement')

# ── Master title ───────────────────────────────────────────
fig2.suptitle('Placement Prediction — Scatter Plot Analysis',
              color=TEXT_C, fontsize=17, fontweight='bold', y=1.005)

scatter_path = os.path.join(base_path, 'scatter_report.png')
plt.savefig(scatter_path, dpi=150, bbox_inches='tight', facecolor=PAGE_BG)
plt.close()
print(f"✅ scatter_report.png saved → {scatter_path}")


# ══════════════════════════════════════════════════════════════
#  REPORT 3  –  Full Pairplot Matrix  (pairplot_report.png)
#  Shows every numeric feature against every other, coloured
#  by Placement_Status
# ══════════════════════════════════════════════════════════════
print("⏳ Building pairplot (may take a moment)...")

plot_df = X_train[numeric_features].copy()
plot_df['Placement'] = y_col.map(dict(enumerate(class_labels)))

pair_palette = {class_labels[1]: PLACED, class_labels[0]: NOTPLACE}

with plt.rc_context({'axes.facecolor': BG,
                     'figure.facecolor': PAGE_BG,
                     'text.color': TEXT_C,
                     'axes.labelcolor': TEXT_C,
                     'xtick.color': TEXT_C,
                     'ytick.color': TEXT_C,
                     'grid.color': GRID_C,
                     'grid.linewidth': 0.5}):

    g = sns.pairplot(
        plot_df,
        hue='Placement',
        palette=pair_palette,
        plot_kws=dict(alpha=0.45, s=14, edgecolor='none'),
        diag_kws=dict(alpha=0.5, linewidth=1.2),
        corner=True,
    )

    g.figure.suptitle('Pairplot — All Features vs Placement Status',
                      color=TEXT_C, fontsize=14, fontweight='bold', y=1.01)
    g.figure.patch.set_facecolor(PAGE_BG)

    for ax in g.axes.flatten():
        if ax is not None:
            ax.set_facecolor(BG)
            for spine in ax.spines.values():
                spine.set_edgecolor(GRID_C)

pairplot_path = os.path.join(base_path, 'pairplot_report.png')
g.figure.savefig(pairplot_path, dpi=130, bbox_inches='tight', facecolor=PAGE_BG)
plt.close()
print(f"✅ pairplot_report.png saved → {pairplot_path}")

print("\n🎉 All reports generated!")
print("   📄 ml_report.png       — original 5-panel model comparison")
print("   📄 scatter_report.png  — 12-panel scatter analysis")
print("   📄 pairplot_report.png — full feature pairplot matrix")
