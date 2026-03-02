"""
=======================================================================
TASK 6: Music Genre Classification - GTZAN Dataset
=======================================================================
- Chargement du dataset GTZAN
- Visualisations complètes (waveforms, spectrogrammes, MFCCs, etc.)
- Extraction de features audio (approche tabulaire)
- Entraînement de modèles ML (Random Forest, SVM, KNN, etc.)
- Approche CNN sur spectrogrammes
- Transfer Learning avec VGG16
- Comparaison finale de toutes les approches
=======================================================================
"""

import os
import sys
import subprocess
import warnings
import time

# ==============================================================================
# Auto-installation des dépendances manquantes  (AVANT tout autre import)
# ==============================================================================
_REQUIRED_PACKAGES = {
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'numpy': 'numpy',
    'pandas': 'pandas',
    'sklearn': 'scikit-learn',
    'librosa': 'librosa',
    'soundfile': 'soundfile',
    'joblib': 'joblib',
    'kagglehub': 'kagglehub',
    'tensorflow': 'tensorflow',
}

print("Vérification des dépendances...")
for _import_name, _pkg_name in _REQUIRED_PACKAGES.items():
    try:
        __import__(_import_name)
        print(f"  ✓ {_import_name} déjà présent")
    except ImportError:
        print(f"  📦 Installation de {_pkg_name}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', _pkg_name, '-q'])
        print(f"  ✓ {_pkg_name} installé !")
print("  ✓ Toutes les dépendances sont prêtes !\n")

# ==============================================================================
# Imports principaux
# ==============================================================================
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour sauvegarder les images
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings('ignore')

# ==============================================================================
# Configuration
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIS_DIR = os.path.join(BASE_DIR, 'visualizations')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
SPEC_DIR = os.path.join(BASE_DIR, 'spectrograms_data')

for d in [VIS_DIR, os.path.join(VIS_DIR, 'waveforms'), os.path.join(VIS_DIR, 'spectrograms'),
          os.path.join(VIS_DIR, 'mfccs'), MODELS_DIR, RESULTS_DIR, SPEC_DIR]:
    os.makedirs(d, exist_ok=True)

print("=" * 70)
print("   TASK 6: MUSIC GENRE CLASSIFICATION - GTZAN DATASET")
print("=" * 70)


# ==============================================================================
# ÉTAPE 1 : Téléchargement du dataset
# ==============================================================================
def step1_download_dataset():
    print("\n" + "=" * 70)
    print("ÉTAPE 1 : Téléchargement du dataset GTZAN")
    print("=" * 70)

    import kagglehub
    path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")
    print(f"Dataset téléchargé : {path}")

    # Trouver le dossier genres_original
    audio_path = None
    for root, dirs, files in os.walk(path):
        if 'genres_original' in dirs:
            audio_path = os.path.join(root, 'genres_original')
            break

    if audio_path is None:
        # Essayer 'genres'
        for root, dirs, files in os.walk(path):
            if 'genres' in dirs:
                audio_path = os.path.join(root, 'genres')
                break

    if audio_path is None:
        print("ERREUR: Impossible de trouver le dossier audio.")
        print("Structure du dataset:")
        for root, dirs, files in os.walk(path):
            level = root.replace(path, '').count(os.sep)
            indent = '  ' * level
            print(f'{indent}{os.path.basename(root)}/')
            if level < 3:
                for f in files[:3]:
                    print(f'{indent}  {f}')
                if len(files) > 3:
                    print(f'{indent}  ... +{len(files)-3} fichiers')
        sys.exit(1)

    print(f"Chemin audio trouvé : {audio_path}")

    # Lister les genres
    genres = sorted([g for g in os.listdir(audio_path)
                     if os.path.isdir(os.path.join(audio_path, g))])
    print(f"\nGenres disponibles ({len(genres)}): {genres}")

    genre_counts = {}
    for genre in genres:
        gp = os.path.join(audio_path, genre)
        cnt = len([f for f in os.listdir(gp) if f.endswith('.wav') or f.endswith('.au')])
        genre_counts[genre] = cnt
        print(f"  {genre}: {cnt} fichiers")

    # Chercher aussi le CSV pré-extrait
    csv_path = None
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith('.csv') and 'feature' in f.lower():
                csv_path = os.path.join(root, f)
                break
            elif f == 'features_30_sec.csv':
                csv_path = os.path.join(root, f)
                break
            elif f == 'features_3_sec.csv' and csv_path is None:
                csv_path = os.path.join(root, f)

    if csv_path:
        print(f"\nCSV de features trouvé : {csv_path}")

    return audio_path, genres, genre_counts, csv_path


# ==============================================================================
# ÉTAPE 2 : Visualisations
# ==============================================================================
def step2_visualizations(audio_path, genres, genre_counts):
    print("\n" + "=" * 70)
    print("ÉTAPE 2 : Visualisations")
    print("=" * 70)

    # --- 2a. Distribution des genres ---
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(genres)))
    bars = plt.bar(genre_counts.keys(), genre_counts.values(), color=colors)
    plt.xlabel('Genre', fontsize=12)
    plt.ylabel('Nombre de fichiers', fontsize=12)
    plt.title('Distribution des fichiers audio par genre', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    for bar, val in zip(bars, genre_counts.values()):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(val), ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'genre_distribution.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ genre_distribution.png")

    # --- 2b. Waveforms, spectrogrammes, MFCCs pour chaque genre ---
    print("\n  Génération des visualisations par genre...")
    for genre in genres:
        gp = os.path.join(audio_path, genre)
        files = sorted([f for f in os.listdir(gp) if f.endswith('.wav') or f.endswith('.au')])
        if not files:
            continue

        fpath = os.path.join(gp, files[0])
        try:
            y, sr = librosa.load(fpath, duration=30)

            # Waveform
            fig, ax = plt.subplots(figsize=(12, 3))
            librosa.display.waveshow(y, sr=sr, ax=ax, alpha=0.7)
            ax.set_title(f'Waveform - {genre}', fontsize=12)
            ax.set_xlabel('Temps (s)')
            ax.set_ylabel('Amplitude')
            plt.tight_layout()
            plt.savefig(os.path.join(VIS_DIR, 'waveforms', f'{genre}_waveform.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()

            # Mel Spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            S_dB = librosa.power_to_db(S, ref=np.max)
            fig, ax = plt.subplots(figsize=(12, 5))
            img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            ax.set_title(f'Mel Spectrogram - {genre}', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(VIS_DIR, 'spectrograms', f'{genre}_spectrogram.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()

            # MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            fig, ax = plt.subplots(figsize=(12, 5))
            img = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax)
            fig.colorbar(img, ax=ax)
            ax.set_title(f'MFCCs - {genre}', fontsize=12)
            ax.set_ylabel('Coefficients MFCC')
            plt.tight_layout()
            plt.savefig(os.path.join(VIS_DIR, 'mfccs', f'{genre}_mfcc.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()

            print(f"    ✓ {genre}")
        except Exception as e:
            print(f"    ✗ {genre} - erreur: {e}")

    # --- 2c. Comparaison de tous les spectrogrammes ---
    n_genres = len(genres)
    ncols = 5
    nrows = (n_genres + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows))
    axes = axes.flatten()

    for idx, genre in enumerate(genres):
        gp = os.path.join(audio_path, genre)
        files = sorted([f for f in os.listdir(gp) if f.endswith('.wav') or f.endswith('.au')])
        if files:
            try:
                y, sr = librosa.load(os.path.join(gp, files[0]), duration=30)
                S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                S_dB = librosa.power_to_db(S, ref=np.max)
                librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=axes[idx])
                axes[idx].set_title(genre.upper(), fontsize=10, fontweight='bold')
                axes[idx].set_xlabel('')
                axes[idx].set_ylabel('')
            except:
                axes[idx].set_title(f'{genre} (erreur)')

    # Masquer les axes vides
    for idx in range(len(genres), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Comparaison des spectrogrammes Mel par genre', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'all_genres_spectrograms_comparison.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ all_genres_spectrograms_comparison.png")


# ==============================================================================
# ÉTAPE 3 : Extraction de features audio
# ==============================================================================
def extract_features_from_file(file_path, duration=30):
    """Extrait les caractéristiques audio d'un fichier."""
    try:
        y, sr = librosa.load(file_path, duration=duration)

        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)

        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)

        # Spectral features
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)
        sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        sr_feat = librosa.feature.spectral_rolloff(y=y, sr=sr)
        scon = librosa.feature.spectral_contrast(y=y, sr=sr)

        zcr = librosa.feature.zero_crossing_rate(y)
        rms = librosa.feature.rms(y=y)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        harmony = np.mean(librosa.effects.harmonic(y))
        perceptr = np.mean(librosa.effects.percussive(y))

        features = np.concatenate([
            mfccs_mean, mfccs_std,                   # 26
            chroma_mean, chroma_std,                  # 24
            [np.mean(sc), np.std(sc)],                # 2
            [np.mean(sb), np.std(sb)],                # 2
            [np.mean(sr_feat), np.std(sr_feat)],      # 2
            np.mean(scon, axis=1),                    # 7
            [np.mean(zcr), np.std(zcr)],              # 2
            [np.mean(rms), np.std(rms)],              # 2
            [float(tempo) if np.isscalar(tempo) else float(tempo[0]) if hasattr(tempo, '__len__') else float(tempo)],  # 1
            [harmony, perceptr]                        # 2
        ])
        return features
    except Exception as e:
        return None


def step3_extract_features(audio_path, genres):
    print("\n" + "=" * 70)
    print("ÉTAPE 3 : Extraction des caractéristiques audio")
    print("=" * 70)

    feature_names = (
        [f'mfcc{i}_mean' for i in range(1, 14)] +
        [f'mfcc{i}_std' for i in range(1, 14)] +
        [f'chroma{i}_mean' for i in range(1, 13)] +
        [f'chroma{i}_std' for i in range(1, 13)] +
        ['spectral_centroid_mean', 'spectral_centroid_std',
         'spectral_bandwidth_mean', 'spectral_bandwidth_std',
         'spectral_rolloff_mean', 'spectral_rolloff_std'] +
        [f'spectral_contrast{i}' for i in range(1, 8)] +
        ['zcr_mean', 'zcr_std', 'rms_mean', 'rms_std',
         'tempo', 'harmony', 'perceptr']
    )

    print(f"  Nombre de features par fichier : {len(feature_names)}")

    data = []
    labels = []
    filenames_list = []
    errors = []

    for genre in genres:
        gp = os.path.join(audio_path, genre)
        files = sorted([f for f in os.listdir(gp) if f.endswith('.wav') or f.endswith('.au')])
        print(f"\n  Genre: {genre} ({len(files)} fichiers)")

        for i, fname in enumerate(files):
            fpath = os.path.join(gp, fname)
            feats = extract_features_from_file(fpath)
            if feats is not None and len(feats) == len(feature_names):
                data.append(feats)
                labels.append(genre)
                filenames_list.append(fname)
            else:
                errors.append((genre, fname))

            if (i + 1) % 25 == 0 or (i + 1) == len(files):
                print(f"    Traité : {i + 1}/{len(files)}")

    print(f"\n  ✓ Extraction terminée !")
    print(f"    Succès : {len(data)}")
    print(f"    Erreurs : {len(errors)}")
    if errors:
        for g, f in errors[:5]:
            print(f"      - {g}/{f}")

    df = pd.DataFrame(data, columns=feature_names)
    df['genre'] = labels
    df['filename'] = filenames_list
    csv_out = os.path.join(RESULTS_DIR, 'audio_features.csv')
    df.to_csv(csv_out, index=False)
    print(f"  Sauvegardé : {csv_out}")

    return df, feature_names


# ==============================================================================
# ÉTAPE 4 : Analyse exploratoire des features
# ==============================================================================
def step4_eda(df):
    print("\n" + "=" * 70)
    print("ÉTAPE 4 : Analyse exploratoire des features")
    print("=" * 70)

    print(f"  Shape : {df.shape}")
    print(f"  Valeurs manquantes : {df.isnull().sum().sum()}")
    print(f"\n  Statistiques :")
    print(df.describe().to_string()[:500])

    # Distribution des MFCC moyens par genre
    mfcc_mean_cols = [c for c in df.columns if 'mfcc' in c and 'mean' in c]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for i, col in enumerate(mfcc_mean_cols[:6]):
        for genre in df['genre'].unique():
            subset = df[df['genre'] == genre][col]
            axes[i].hist(subset, alpha=0.5, label=genre, bins=15)
        axes[i].set_title(col)
        axes[i].set_xlabel('Valeur')
        axes[i].set_ylabel('Fréquence')
        if i == 0:
            axes[i].legend(fontsize=6, loc='upper right')
    plt.suptitle('Distribution des MFCCs par genre', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'mfcc_distributions.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ mfcc_distributions.png")

    # Boxplots des features spectrales
    spectral_cols = ['spectral_centroid_mean', 'spectral_bandwidth_mean',
                     'spectral_rolloff_mean', 'tempo', 'zcr_mean', 'rms_mean']
    existing = [c for c in spectral_cols if c in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for idx, col in enumerate(existing[:6]):
        df.boxplot(column=col, by='genre', ax=axes[idx])
        axes[idx].set_title(col)
        axes[idx].set_xlabel('')
        axes[idx].tick_params(axis='x', rotation=45)
    for idx in range(len(existing), 6):
        axes[idx].set_visible(False)
    plt.suptitle('Caractéristiques spectrales par genre', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'spectral_features_boxplot.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ spectral_features_boxplot.png")

    # Matrice de corrélation
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    plt.figure(figsize=(18, 15))
    sns.heatmap(corr, cmap='coolwarm', center=0, linewidths=0.1,
                cbar_kws={'shrink': 0.6}, square=True)
    plt.title('Matrice de corrélation des features', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'correlation_matrix.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ correlation_matrix.png")

    # Pairplot d'un sous-ensemble de features
    subset_cols = ['mfcc1_mean', 'mfcc2_mean', 'spectral_centroid_mean', 'tempo', 'genre']
    existing_sub = [c for c in subset_cols if c in df.columns]
    if len(existing_sub) >= 3:
        sns.pairplot(df[existing_sub], hue='genre', diag_kind='kde',
                     plot_kws={'alpha': 0.5, 's': 20})
        plt.suptitle('Pairplot de features sélectionnées', y=1.02, fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(VIS_DIR, 'pairplot_features.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ pairplot_features.png")

    # Mean MFCC heatmap par genre
    mfcc_mean_all = [c for c in df.columns if 'mfcc' in c and 'mean' in c]
    genre_mfcc = df.groupby('genre')[mfcc_mean_all].mean()
    plt.figure(figsize=(14, 8))
    sns.heatmap(genre_mfcc, annot=True, fmt='.1f', cmap='YlOrRd', linewidths=0.5)
    plt.title('Moyenne des MFCCs par genre', fontsize=14)
    plt.ylabel('Genre')
    plt.xlabel('MFCC Coefficient')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'mfcc_heatmap_by_genre.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ mfcc_heatmap_by_genre.png")


# ==============================================================================
# ÉTAPE 5 : Modèles tabulaires (Scikit-learn)
# ==============================================================================
def step5_tabular_models(df):
    print("\n" + "=" * 70)
    print("ÉTAPE 5 : Entraînement des modèles tabulaires (Scikit-learn)")
    print("=" * 70)

    X = df.drop(['genre', 'filename'], axis=1, errors='ignore')
    y = df['genre']

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    print(f"  X_train : {X_train.shape}, X_test : {X_test.shape}")
    print(f"  Classes : {le.classes_}")

    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    joblib.dump(le, os.path.join(MODELS_DIR, 'label_encoder.pkl'))

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(max_iter=2000, random_state=42),
        'MLP Neural Net': MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=500, random_state=42),
    }

    results = {}

    for name, model in models.items():
        print(f"\n  {'─' * 50}")
        print(f"  Modèle : {name}")
        print(f"  {'─' * 50}")

        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        results[name] = {
            'accuracy': acc,
            'time': train_time,
            'y_pred': y_pred,
        }

        print(f"  Accuracy : {acc:.4f}  |  Temps : {train_time:.2f}s")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        joblib.dump(model, os.path.join(MODELS_DIR, f'{name.lower().replace(" ", "_").replace("(", "").replace(")", "")}.pkl'))

    # --- Résumé ---
    print("\n  " + "=" * 50)
    print("  RÉSUMÉ DES MODÈLES TABULAIRES")
    print("  " + "=" * 50)
    summary = pd.DataFrame([
        {'Modèle': k, 'Accuracy': v['accuracy'], 'Temps (s)': round(v['time'], 2)}
        for k, v in results.items()
    ]).sort_values('Accuracy', ascending=False)
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(RESULTS_DIR, 'tabular_model_comparison.csv'), index=False)

    # Graphique de comparaison
    plt.figure(figsize=(12, 6))
    colors_bar = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(summary)))
    bars = plt.barh(summary['Modèle'], summary['Accuracy'], color=colors_bar)
    plt.xlabel('Accuracy', fontsize=12)
    plt.title('Comparaison des modèles tabulaires', fontsize=14)
    plt.xlim(0, 1)
    for bar, acc in zip(bars, summary['Accuracy']):
        plt.text(acc + 0.01, bar.get_y() + bar.get_height() / 2,
                 f'{acc:.3f}', va='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'tabular_model_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ tabular_model_comparison.png")

    # Matrice de confusion du meilleur modèle
    best_name = max(results, key=lambda k: results[k]['accuracy'])
    best_pred = results[best_name]['y_pred']
    best_acc = results[best_name]['accuracy']

    cm = confusion_matrix(y_test, best_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Prédiction', fontsize=12)
    plt.ylabel('Vrai label', fontsize=12)
    plt.title(f'Matrice de confusion - {best_name} (Acc={best_acc:.3f})', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'best_tabular_confusion_matrix.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ best_tabular_confusion_matrix.png (Meilleur : {best_name})")

    # Matrice normalisée
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Prédiction', fontsize=12)
    plt.ylabel('Vrai label', fontsize=12)
    plt.title(f'Matrice de confusion normalisée - {best_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'best_tabular_confusion_matrix_norm.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ best_tabular_confusion_matrix_norm.png")

    return results, le, best_name, best_acc


# ==============================================================================
# ÉTAPE 6 : Génération d'images de spectrogrammes pour CNN
# ==============================================================================
def step6_generate_spectrogram_images(audio_path, genres):
    print("\n" + "=" * 70)
    print("ÉTAPE 6 : Génération d'images de spectrogrammes pour CNN")
    print("=" * 70)

    for split in ['train', 'test']:
        for genre in genres:
            os.makedirs(os.path.join(SPEC_DIR, split, genre), exist_ok=True)

    total_train = 0
    total_test = 0

    for genre in genres:
        gp = os.path.join(audio_path, genre)
        files = sorted([f for f in os.listdir(gp) if f.endswith('.wav') or f.endswith('.au')])

        train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)

        for flist, split in [(train_files, 'train'), (test_files, 'test')]:
            for fname in flist:
                src = os.path.join(gp, fname)
                base = os.path.splitext(fname)[0]
                dst = os.path.join(SPEC_DIR, split, genre, f'{base}.png')

                if os.path.exists(dst):
                    if split == 'train':
                        total_train += 1
                    else:
                        total_test += 1
                    continue

                try:
                    y, sr = librosa.load(src, duration=30)
                    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                    S_dB = librosa.power_to_db(S, ref=np.max)

                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.axis('off')
                    librosa.display.specshow(S_dB, sr=sr, ax=ax)
                    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                    plt.savefig(dst, bbox_inches='tight', pad_inches=0, dpi=72)
                    plt.close()

                    if split == 'train':
                        total_train += 1
                    else:
                        total_test += 1
                except Exception as e:
                    pass

        print(f"  ✓ {genre}")

    print(f"\n  Total images : train={total_train}, test={total_test}")


# ==============================================================================
# ÉTAPE 7 : CNN sur spectrogrammes
# ==============================================================================
def step7_cnn(genres):
    print("\n" + "=" * 70)
    print("ÉTAPE 7 : Entraînement du CNN sur spectrogrammes")
    print("=" * 70)

    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers, models as keras_models
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    except ImportError:
        print("  ⚠ TensorFlow non disponible. Passage au CNN ignoré.")
        return None, None

    IMG_SIZE = (128, 128)
    BATCH_SIZE = 32
    EPOCHS = 40

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        validation_split=0.2
    )

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_dir = os.path.join(SPEC_DIR, 'train')
    test_dir = os.path.join(SPEC_DIR, 'test')

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='training', shuffle=True
    )

    val_gen = train_datagen.flow_from_directory(
        train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='validation', shuffle=True
    )

    test_gen = test_datagen.flow_from_directory(
        test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=False
    )

    num_classes = len(train_gen.class_indices)
    print(f"  Classes : {train_gen.class_indices}")
    print(f"  Train samples : {train_gen.samples}")
    print(f"  Val samples   : {val_gen.samples}")
    print(f"  Test samples  : {test_gen.samples}")

    # Construire le CNN
    model = keras_models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(*IMG_SIZE, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6),
        ModelCheckpoint(os.path.join(MODELS_DIR, 'cnn_best.keras'),
                        monitor='val_accuracy', save_best_only=True, mode='max')
    ]

    print("\n  Entraînement du CNN...")
    history = model.fit(
        train_gen, epochs=EPOCHS, validation_data=val_gen,
        callbacks=callbacks, verbose=1
    )

    model.save(os.path.join(MODELS_DIR, 'cnn_final.keras'))

    # Courbes d'entraînement
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0].set_title('CNN - Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[1].set_title('CNN - Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'cnn_training_history.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ cnn_training_history.png")

    # Évaluation
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"\n  CNN Test Accuracy : {test_acc:.4f}")

    test_gen.reset()
    y_pred_proba = model.predict(test_gen)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = test_gen.classes
    class_names = list(test_gen.class_indices.keys())

    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Prédiction')
    plt.ylabel('Vrai label')
    plt.title(f'Matrice de confusion - CNN (Acc={test_acc:.3f})')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'cnn_confusion_matrix.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ cnn_confusion_matrix.png")

    return test_acc, model


# ==============================================================================
# ÉTAPE 8 : Transfer Learning (VGG16)
# ==============================================================================
def step8_transfer_learning(genres):
    print("\n" + "=" * 70)
    print("ÉTAPE 8 : Transfer Learning avec VGG16")
    print("=" * 70)

    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        from tensorflow.keras.applications import VGG16
        from tensorflow.keras.models import Model
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    except ImportError:
        print("  ⚠ TensorFlow non disponible.")
        return None

    IMG_SIZE = (128, 128)
    BATCH_SIZE = 32
    EPOCHS = 30

    train_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_dir = os.path.join(SPEC_DIR, 'train')
    test_dir = os.path.join(SPEC_DIR, 'test')

    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='training'
    )
    val_gen = train_datagen.flow_from_directory(
        train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='validation'
    )
    test_gen = test_datagen.flow_from_directory(
        test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=False
    )

    num_classes = len(train_gen.class_indices)

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(*IMG_SIZE, 3))
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    preds = layers.Dense(num_classes, activation='softmax')(x)

    transfer_model = Model(inputs=base_model.input, outputs=preds)

    transfer_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6),
        ModelCheckpoint(os.path.join(MODELS_DIR, 'transfer_best.keras'),
                        monitor='val_accuracy', save_best_only=True, mode='max')
    ]

    print("\n  Entraînement Transfer Learning (VGG16)...")
    history = transfer_model.fit(
        train_gen, epochs=EPOCHS, validation_data=val_gen,
        callbacks=callbacks, verbose=1
    )

    transfer_model.save(os.path.join(MODELS_DIR, 'transfer_final.keras'))

    # Courbes
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0].set_title('VGG16 Transfer - Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[1].set_title('VGG16 Transfer - Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'transfer_training_history.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ transfer_training_history.png")

    # Évaluation
    test_loss, test_acc = transfer_model.evaluate(test_gen)
    print(f"\n  VGG16 Test Accuracy : {test_acc:.4f}")

    test_gen.reset()
    y_pred = np.argmax(transfer_model.predict(test_gen), axis=1)
    y_true = test_gen.classes
    class_names = list(test_gen.class_indices.keys())

    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Prédiction')
    plt.ylabel('Vrai label')
    plt.title(f'Matrice de confusion - VGG16 Transfer (Acc={test_acc:.3f})')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'transfer_confusion_matrix.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ transfer_confusion_matrix.png")

    return test_acc


# ==============================================================================
# ÉTAPE 9 : Comparaison finale
# ==============================================================================
def step9_final_comparison(tabular_results, best_tabular_name, best_tabular_acc,
                           cnn_acc, transfer_acc):
    print("\n" + "=" * 70)
    print("ÉTAPE 9 : Comparaison finale de toutes les approches")
    print("=" * 70)

    rows = [
        {'Approche': f'Tabulaire - {best_tabular_name}', 'Accuracy': best_tabular_acc,
         'Type': 'Features audio'},
    ]

    if cnn_acc is not None:
        rows.append({'Approche': 'CNN Custom', 'Accuracy': cnn_acc, 'Type': 'Spectrogrammes'})
    if transfer_acc is not None:
        rows.append({'Approche': 'Transfer Learning (VGG16)', 'Accuracy': transfer_acc,
                     'Type': 'Spectrogrammes'})

    final_df = pd.DataFrame(rows).sort_values('Accuracy', ascending=False)
    print("\n" + final_df.to_string(index=False))
    final_df.to_csv(os.path.join(RESULTS_DIR, 'final_comparison.csv'), index=False)

    # Graphique
    plt.figure(figsize=(10, 6))
    colors_list = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    bars = plt.bar(final_df['Approche'], final_df['Accuracy'],
                   color=colors_list[:len(final_df)])
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Comparaison finale : Tabulaire vs CNN vs Transfer Learning', fontsize=14)
    plt.ylim(0, 1)
    for bar, acc in zip(bars, final_df['Accuracy']):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{acc:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, 'final_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("  ✓ final_comparison.png")

    # Afficher le récapitulatif
    print("\n" + "=" * 70)
    print("  RÉCAPITULATIF DES FICHIERS GÉNÉRÉS")
    print("=" * 70)

    for folder in [VIS_DIR, MODELS_DIR, RESULTS_DIR]:
        print(f"\n  📁 {os.path.basename(folder)}/")
        for root, dirs, files in os.walk(folder):
            level = root.replace(folder, '').count(os.sep)
            indent = '    ' + '  ' * level
            if level > 0:
                print(f'{indent}📁 {os.path.basename(root)}/')
            for f in sorted(files):
                print(f'{indent}  📄 {f}')

    print("\n" + "=" * 70)
    print("  ✅ TÂCHE TERMINÉE AVEC SUCCÈS !")
    print("=" * 70)


# ==============================================================================
# MAIN  – Reprend à partir de l'étape 5 (features CSV déjà généré)
# ==============================================================================
if __name__ == '__main__':
    start_total = time.time()

    # ── Chemins dataset (nécessaires pour étapes 6-8) ──────────────────────────
    print("\n" + "=" * 70)
    print("   Reprise à partir de l'ÉTAPE 5 (CSV déjà extrait)")
    print("=" * 70)

    # Retrouver le chemin audio via kagglehub (pas de re-téléchargement si déjà en cache)
    import kagglehub
    _dl_path = kagglehub.dataset_download("andradaolteanu/gtzan-dataset-music-genre-classification")
    audio_path = None
    for _root, _dirs, _files in os.walk(_dl_path):
        if 'genres_original' in _dirs:
            audio_path = os.path.join(_root, 'genres_original')
            break
    if audio_path is None:
        for _root, _dirs, _files in os.walk(_dl_path):
            if 'genres' in _dirs:
                audio_path = os.path.join(_root, 'genres')
                break
    if audio_path is None:
        print("ERREUR : impossible de localiser le dossier audio – vérifiez le cache kagglehub.")
        sys.exit(1)

    genres = sorted([g for g in os.listdir(audio_path)
                     if os.path.isdir(os.path.join(audio_path, g))])
    print(f"  Dataset audio : {audio_path}")
    print(f"  Genres ({len(genres)}) : {genres}")

    # ── Charger le CSV de features déjà extrait ────────────────────────────────
    csv_features = os.path.join(RESULTS_DIR, 'audio_features.csv')
    if not os.path.exists(csv_features):
        print(f"ERREUR : {csv_features} introuvable. Lancez d'abord les étapes 1-4.")
        sys.exit(1)

    df = pd.read_csv(csv_features)
    print(f"  ✓ CSV chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")

    # ── Étape 5 : Modèles tabulaires ───────────────────────────────────────────
    tabular_results, le, best_tabular_name, best_tabular_acc = step5_tabular_models(df)

    # ── Étape 6 : Générer les images de spectrogrammes ─────────────────────────
    step6_generate_spectrogram_images(audio_path, genres)

    # ── Étape 7 : CNN ──────────────────────────────────────────────────────────
    cnn_acc, cnn_model = step7_cnn(genres)

    # ── Étape 8 : Transfer Learning ────────────────────────────────────────────
    transfer_acc = step8_transfer_learning(genres)

    # ── Étape 9 : Comparaison finale ───────────────────────────────────────────
    step9_final_comparison(tabular_results, best_tabular_name, best_tabular_acc,
                           cnn_acc, transfer_acc)

    total_time = time.time() - start_total
    print(f"\n  ⏱ Temps total d'exécution : {total_time / 60:.1f} minutes")

