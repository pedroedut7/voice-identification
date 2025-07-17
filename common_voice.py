#!/usr/bin/env python3
"""
Voice Identification System Using FFT - Common Voice Dataset Pipeline
Sistema de Identificação de Voz usando FFT - Pipeline com Dataset Common Voice

Implementa uma pipeline completa para identificação de voz utilizando:
- Dataset Common Voice (pt-BR)
- Extração de características usando FFT e MFCC
- Classificação com Machine Learning
- Tratamento para dataset desbalanceado (filtragem de locutores)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import warnings
import os
import tarfile

warnings.filterwarnings('ignore')

class VoiceIdentificationPipeline:
    """Pipeline principal para identificação de voz com dataset Common Voice"""
    
    EXPECTED_NUM_FEATURES = 0
    
    def __init__(self, sample_rate=16000, features_to_use=[1,1,1,1,1,1], n_fft=2048, hop_length=512):
        """
        Inicializa a pipeline de identificação de voz
        
        Args:
            sample_rate (int): Taxa de amostragem do áudio
            n_fft (int): Tamanho da janela FFT
            hop_length (int): Tamanho do passo para análise
        """
        self.sample_rate = sample_rate
        self.features_to_use = features_to_use
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.features = None
        self.labels = None
        self.audio_data_cache = {} # Cache para armazenar áudios carregados
        self.calculate_exepcted_number_of_features()

    def calculate_exepcted_number_of_features(self):
        self.EXPECTED_NUM_FEATURES += self.features_to_use[0] * 3 + self.features_to_use[1] * 5 + self.features_to_use[2] * 5
        self.EXPECTED_NUM_FEATURES += self.features_to_use[3] * 52 + self.features_to_use[4] * 104 + self.features_to_use[5] * 2 
        
    def extract_common_voice_dataset(self, tar_gz_path, extract_path='cv-corpus'):
        """
        Extrai o dataset Common Voice do arquivo .tar.gz
        
        Args:
            tar_gz_path (str): Caminho para o arquivo .tar.gz do Common Voice.
            extract_path (str): Diretório para onde os arquivos serão extraídos.
        
        Returns:
            str: Caminho completo para o diretório de dados extraídos.
        """
        if not os.path.exists(extract_path):
            os.makedirs(extract_path, exist_ok=True)
            try:
                with tarfile.open(tar_gz_path, "r:gz") as tar:
                    tar.extractall(path=extract_path)
                print(f"Dataset extraído com sucesso para {extract_path}")
            except tarfile.ReadError as e:
                print(f"Erro ao ler o arquivo tar.gz: {e}")
                print("Certifique-se de que o arquivo está íntegro e é um tar.gz válido.")
                return None
            except Exception as e:
                print(f"Erro inesperado ao extrair o dataset: {e}")
                return None
        else:
            print(f"Diretório de extração '{extract_path}' já existe. Pulando a extração.")
        
        return extract_path

    def load_common_voice_dataset(self, dataset_base_path, max_samples=None, split_file='validated.tsv', min_samples_per_speaker=5, max_samples_per_speaker=None):
        """
        Carrega o dataset Common Voice de arquivos locais e aplica filtragem por locutor.
        
        Args:
            dataset_base_path (str): Caminho base onde o dataset Common Voice foi extraído.
            max_samples (int, optional): Número máximo de amostras para carregar ANTES da filtragem por locutor.
            split_file (str): Nome do arquivo TSV a ser usado (ex: 'validated.tsv', 'train.tsv').
            min_samples_per_speaker (int): Número mínimo de amostras que um locutor deve ter para ser incluído.
            max_samples_per_speaker (int, optional): Número máximo de amostras a serem mantidas por locutor (subamostragem).
            
        Returns:
            pd.DataFrame: DataFrame contendo os metadados do dataset filtrado.
        """
        tsv_path = os.path.join(dataset_base_path, split_file)
        if not os.path.exists(tsv_path):
            print(f"Erro: Arquivo TSV não encontrado em {tsv_path}")
            print("Certifique-se de que o dataset foi extraído corretamente e o 'split_file' está correto.")
            return None
        
        try:
            df = pd.read_csv(tsv_path, sep='\t')
            print(f"Metadados carregados com sucesso! Número total de amostras: {len(df)}")
            
            # Limita o número de amostras se especificado (antes de qualquer filtragem mais complexa)
            if max_samples and len(df) > max_samples:
                print(f"Limitando o carregamento inicial a {max_samples} amostras.")
                df = df.head(max_samples)
            
            # Filtra amostras sem client_id válido ou sem path
            df = df.dropna(subset=['path', 'client_id'])
            df = df[df['client_id'] != ''] # Garante que client_id não esteja vazio
            
            print(f"Número de amostras válidas após limpeza inicial: {len(df)}")
            print(f"Número de locutores únicos após limpeza inicial: {df['client_id'].nunique()}")

            # --- Aplicação da Estratégia de Filtragem de Locutores ---
            print(f"\nAplicando filtragem: Mantendo locutores com pelo menos {min_samples_per_speaker} amostras...")
            
            # Conta as amostras por locutor
            speaker_counts = df['client_id'].value_counts()
            
            # Filtra locutores que não atendem ao mínimo de amostras
            qualified_speakers = speaker_counts[speaker_counts >= min_samples_per_speaker].index
            df_filtered = df[df['client_id'].isin(qualified_speakers)].copy()
            
            # Opcional: Subamostragem para locutores com muitas amostras
            if max_samples_per_speaker:
                print(f"Aplicando subamostragem: Limitando a {max_samples_per_speaker} amostras por locutor.")
                df_resampled = pd.DataFrame()
                for speaker_id in qualified_speakers:
                    speaker_samples = df_filtered[df_filtered['client_id'] == speaker_id]
                    if len(speaker_samples) > max_samples_per_speaker:
                        df_resampled = pd.concat([df_resampled, speaker_samples.sample(n=max_samples_per_speaker, random_state=42)])
                    else:
                        df_resampled = pd.concat([df_resampled, speaker_samples])
                df_filtered = df_resampled.reset_index(drop=True)
            
            print(f"Número final de amostras após filtragem: {len(df_filtered)}")
            print(f"Número final de locutores únicos: {df_filtered['client_id'].nunique()}")
            print(f"Novas classes (locutores) e suas contagens:\n{df_filtered['client_id'].value_counts().head()}") # Mostra os primeiros
            
            if df_filtered.empty:
                print("Aviso: Nenhuma amostra válida permaneceu após a filtragem. Ajuste os parâmetros de filtragem.")
                return None

            return df_filtered
            
        except Exception as e:
            print(f"Erro ao carregar ou filtrar metadados do Common Voice: {e}")
            return None
    
    def preprocess_audio(self, audio_data, target_length=None):
        """
        Pré-processa os dados de áudio.
        
        Args:
            audio_data (np.array): Dados de áudio.
            target_length (int): Comprimento alvo para normalização (padding/truncagem).
            
        Returns:
            np.array: Áudio pré-processado.
        """
        # Normaliza o áudio para o range [-1, 1]
        audio_data = audio_data.astype(np.float32)
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Aplica filtro passa-alta para remover ruído de baixa frequência
        sos = signal.butter(3, 80, btype='high', fs=self.sample_rate, output='sos')
        audio_data = signal.sosfilt(sos, audio_data)
        
        # Normaliza o comprimento se especificado
        if target_length:
            if len(audio_data) > target_length:
                audio_data = audio_data[:target_length]
            elif len(audio_data) < target_length:
                audio_data = np.pad(audio_data, (0, target_length - len(audio_data)))
        
        return audio_data
    
    def extract_fft_features(self, audio_data):
        """
        Extrai características usando FFT e MFCCs.

        Args:
            audio_data (np.array): Dados de áudio.

        Returns:
            np.array: Características extraídas.
        """

        USE_FTT_SPECTRAL_FEATURES = self.features_to_use[0]
        USE_SPECTRAL_STATS = self.features_to_use[1]
        USE_FREQUENCY_BANDS = self.features_to_use[2]
        USE_MFCC = self.features_to_use[3]
        USE_MFCC_DELTA = self.features_to_use[4]
        USE_ZERO_CROSS_RATE = self.features_to_use[5]
        features = []

        # 1. FFT Spectral Features
        # Garante que audio_data não esteja vazio para FFT
        if len(audio_data) == 0:
            return np.zeros(171) # Tamanho total das características (ajustado para incluir deltas)

        if USE_FTT_SPECTRAL_FEATURES:
            try:
                fft_spectrum = np.abs(fft(audio_data, n=self.n_fft))
                fft_spectrum = fft_spectrum[:self.n_fft//2]  # Apenas metade positiva

                # Evita divisão por zero para np.sum(fft_spectrum)
                if np.sum(fft_spectrum) == 0:
                    spectral_centroid = 0
                    spectral_rolloff = 0
                    spectral_bandwidth = 0
                else:
                    spectral_centroid = np.sum(fft_spectrum * np.arange(len(fft_spectrum))) / np.sum(fft_spectrum)
                    spectral_rolloff = np.percentile(fft_spectrum, 85)
                    spectral_bandwidth = np.sqrt(np.sum(((np.arange(len(fft_spectrum)) - spectral_centroid) ** 2) * fft_spectrum) / np.sum(fft_spectrum))

                fft_spectral_features = [spectral_centroid, spectral_rolloff, spectral_bandwidth] 
                features.extend(fft_spectral_features)
            except:
                # print(f"Erro na extração de características FFT: {e}. Usando zeros.") # Descomentar para debug
                features.extend([0] * 3) # 3 espectrais + 5 estatísticas + 5 bandas

        if USE_SPECTRAL_STATS:
            fft_spectrum = np.abs(fft(audio_data, n=self.n_fft))
            fft_spectrum = fft_spectrum[:self.n_fft//2]  # Apenas metade positiva

            try:
                # 2. Características estatísticas do espectro
                spectral_stats = [
                    np.mean(fft_spectrum),
                    np.std(fft_spectrum),
                    np.max(fft_spectrum),
                    np.min(fft_spectrum),
                    np.median(fft_spectrum)
                ]
                features.extend(spectral_stats)
            except:
                # print(f"Erro na extração de características FFT: {e}. Usando zeros.") # Descomentar para debug
                features.extend([0] * 5) # 3 espectrais + 5 estatísticas + 5 bandas
                

            # 3. Energia em bandas de frequência específicas
        if USE_FREQUENCY_BANDS:
            fft_spectrum = np.abs(fft(audio_data, n=self.n_fft))
            fft_spectrum = fft_spectrum[:self.n_fft//2]  # Apenas metade positiva

            try:
                freqs = fftfreq(self.n_fft, 1/self.sample_rate)[:self.n_fft//2]

                # Bandas de frequência para análise da fala
                bands = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
                for low, high in bands:
                    band_mask = (freqs >= low) & (freqs <= high)
                    band_energy = np.sum(fft_spectrum[band_mask])
                    features.append(band_energy)
            except Exception as e:
                # print(f"Erro na extração de características FFT: {e}. Usando zeros.") # Descomentar para debug
                features.extend([0] * 5) # 3 espectrais + 5 estatísticas + 5 bandas

        # 4. MFCC, Delta-MFCC e Delta-Delta-MFCC usando librosa (complementa FFT)
        # MFCCs exigem áudio não vazio
        if USE_MFCC:    
            try:
                if len(audio_data) > 0:
                    # Calcula os MFCCs
                    mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)

                    # Combina as características estatísticas para MFCCs, Delta e Delta-Delta
                    mfcc_features_base = [
                        np.mean(mfccs, axis=1),
                        np.std(mfccs, axis=1),
                        np.max(mfccs, axis=1),
                        np.min(mfccs, axis=1)
                    ]
                    mfcc_features_processed = np.concatenate(mfcc_features_base).flatten() 
                    features.extend(mfcc_features_processed)

                    # Calcula os Delta-MFCCs (primeira derivada)
                    delta_mfccs = librosa.feature.delta(mfccs)
                else:
                    # 13 MFCCs * 4 estatísticas (base) + 13 MFCCs * 4 estatísticas (delta) + 13 MFCCs * 4 estatísticas (delta-delta)
                    features.extend([0] * 52) # Total de 156 características para MFCCs e suas derivadas
            except Exception as e:
                # print(f"Erro na extração de MFCCs e derivadas: {e}. Usando zeros.") # Descomentar para debug
                features.extend([0] * 52) # Total de 156 características para MFCCs e suas derivadas

        if USE_MFCC_DELTA:
            try:
                if len(audio_data) > 0:
                    mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
                    delta_mfccs = librosa.feature.delta(mfccs)
                    # Calcula os Delta-Delta-MFCCs (segunda derivada)
                    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
                    mfcc_features_delta = [
                        np.mean(delta_mfccs, axis=1),
                        np.std(delta_mfccs, axis=1),
                        np.max(delta_mfccs, axis=1),
                        np.min(delta_mfccs, axis=1)
                    ]
                    mfcc_features_delta_processed = np.concatenate(mfcc_features_delta).flatten() 
                    features.extend(mfcc_features_delta_processed)

                    mfcc_features_delta2 = [
                        np.mean(delta2_mfccs, axis=1),
                        np.std(delta2_mfccs, axis=1),
                        np.max(delta2_mfccs, axis=1),
                        np.min(delta2_mfccs, axis=1)
                    ]
                    mfcc_features_delta2_processed = np.concatenate(mfcc_features_delta2).flatten() 
                    features.extend(mfcc_features_delta2_processed)
                else:
                    features.extend([0] * (52 + 52))
            except Exception as e:
                # print(f"Erro na extração de MFCCs e derivadas: {e}. Usando zeros.") # Descomentar para debug
                features.extend([0] * (52 + 52))

        # 5. Zero Crossing Rate
        if USE_ZERO_CROSS_RATE:
            try:
                if len(audio_data) > 0:
                    zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
                    zcr_features = [np.mean(zcr), np.std(zcr)]
                    features.extend(zcr_features)
                else:
                    features.extend([0] * 2)
            except Exception as e:
                # print(f"Erro na extração de ZCR: {e}. Usando zeros.") # Descomentar para debug
                features.extend([0] * 2)

        return np.array(features)
    
    def process_dataset(self, df_metadata, dataset_base_path): # Removido max_samples, pois já foi aplicado na carga
        """
        Processa todo o dataset Common Voice extraindo características e labels de locutor.
        
        Args:
            df_metadata (pd.DataFrame): DataFrame com os metadados do dataset (já filtrado).
            dataset_base_path (str): Caminho base do diretório Common Voice.
            
        Returns:
            tuple: (features, labels)
        """
        
        AUDIO_SAMPLES_FOLDER = 'clips'
        
        features_list = []
        labels_list = []
        processed_count = 0
        
        total_samples = len(df_metadata)
        audio_cache = {}
        
        for i, row in df_metadata.iterrows():
            audio_path = os.path.join(dataset_base_path, AUDIO_SAMPLES_FOLDER, row['path'])
            speaker_id = str(row['client_id']) # O client_id é o nosso label de locutor
            
            try:
                if audio_path in audio_cache:
                    audio_data = audio_cache[audio_path]
                else:
                    # Carrega o áudio usando librosa, que é mais robusto
                    audio_data, sr = librosa.load(audio_path, sr=self.sample_rate)
                    audio_cache[audio_path] = audio_data # Armazena no cache
                    
                if len(audio_data) == 0:
                    # print(f"Aviso: Áudio vazio em {audio_path}, pulando.") # Descomentar para debug
                    continue
                
                # Extrai características
                processed_audio = self.preprocess_audio(audio_data)
                features = self.extract_fft_features(processed_audio)
                
                if features.shape[0] != self.EXPECTED_NUM_FEATURES: 
                    print(f"Aviso: Número inesperado de características ({features.shape[0]}) para {audio_path}. Pulando.") # Descomentar para debug
                    continue

                features_list.append(features)
                labels_list.append(speaker_id)
                processed_count += 1
                
                if processed_count % 500 == 0: # Ajustei o print de progresso
                    print(f"Processadas {processed_count}/{total_samples} amostras...")
                    
            except Exception as e:
                print(f"Erro ao processar amostra {audio_path}: {e}")
                continue
        
        print(f"\nProcessamento concluído: {processed_count} amostras válidas")
        
        if len(features_list) == 0:
            raise ValueError("Nenhuma amostra válida foi processada. Verifique os caminhos dos arquivos, o conteúdo do dataset e os parâmetros de filtragem.")
        
        # Converte para arrays numpy e ajusta o formato
        features_array = np.array(features_list)
        labels_array = np.array(labels_list)
        
        return features_array, labels_array
    
    def train_model(self, features, labels, test_size=0.2, model_type='random_forest'):
        """
        Treina o modelo de classificação.
        
        Args:
            features (np.array): Características extraídas.
            labels (np.array): Labels correspondentes (IDs dos locutores).
            test_size (float): Proporção dos dados para teste.
            model_type (str): Tipo de modelo ('random_forest', 'svm').
            
        Returns:
            tuple: (modelo_treinado, acurácia_teste)
        """
        print(f"\nTreinando modelo {model_type}...")
        
        # Verifica se há classes suficientes para treinar
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            print(f"Aviso: Apenas {len(unique_labels)} locutor(es) único(s) encontrado(s). Mínimo de 2 locutores para treinamento.")
            return None, 0.0
        
        # Codifica os labels de locutor
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Divisão treino/teste
        # Stratify é crucial para manter a proporção das classes (locutores) em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            features, encoded_labels, test_size=test_size, random_state=42, stratify=encoded_labels
        )
        
        print(f"Shape do X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"Shape do X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        # Normalização das características
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Escolha do modelo
        if model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced') # Adicionei class_weight
        elif model_type == 'svm':
            model = SVC(kernel='rbf', random_state=42, probability=True, class_weight='balanced') # Adicionei class_weight
        else:
            raise ValueError("Tipo de modelo não suportado. Escolha 'random_forest' ou 'svm'.")
        
        # Treinamento
        model.fit(X_train_scaled, y_train)
        
        # Avaliação
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Acurácia do modelo: {accuracy:.4f}")
        
        # Relatório detalhado
        print("\nRelatório de Classificação:")
        # Para evitar erro com target_names se y_test tiver menos classes que o encoder
        target_names_subset = self.label_encoder.inverse_transform(np.unique(y_test))
        print(classification_report(y_test, y_pred, target_names=target_names_subset))
        
        self.model = model
        return model, accuracy
    
    def predict_audio(self, audio_filepath):
        """
        Prediz a identidade do locutor de um arquivo de áudio.
        
        Args:
            audio_filepath (str): Caminho para o arquivo de áudio.
            
        Returns:
            tuple: (classe_predita, probabilidade)
        """
        if self.model is None:
            raise ValueError("Modelo não foi treinado ainda. Treine o modelo antes de fazer predições.")
        
        try:
            audio_data, sr = librosa.load(audio_filepath, sr=self.sample_rate)
            if sr != self.sample_rate:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.sample_rate)
        except Exception as e:
            print(f"Erro ao carregar o arquivo de áudio para predição: {e}")
            return None, 0.0

        # Pré-processa o áudio
        processed_audio = self.preprocess_audio(audio_data)
        
        # Extrai características
        features = self.extract_fft_features(processed_audio)
        features = features.reshape(1, -1) # Redimensiona para 1 amostra
        
        # Normaliza
        features_scaled = self.scaler.transform(features)
        
        # Aplica PCA se foi usado no treinamento
        if hasattr(self, 'pca'):
            features_scaled = self.pca.transform(features_scaled)
        
        # Predição
        prediction_encoded = self.model.predict(features_scaled)[0]
        probability = np.max(self.model.predict_proba(features_scaled))
        
        # Decodifica o label
        predicted_label = self.label_encoder.inverse_transform([prediction_encoded])[0]
        
        return predicted_label, probability
    
    def visualize_features(self, features, labels, n_samples=300):
        """
        Visualiza as características extraídas usando PCA.
        
        Args:
            features (np.array): Características.
            labels (np.array): Labels (IDs dos locutores).
            n_samples (int): Número máximo de amostras para visualizar.
        """
        if len(features) < 2 or len(np.unique(labels)) < 2:
            print("Não há amostras suficientes ou classes únicas para visualização PCA.")
            return

        print(f"Visualizando características (PCA) para {min(n_samples, len(features))} amostras...")
        
        # Seleciona uma amostra aleatória
        indices = np.random.choice(len(features), min(n_samples, len(features)), replace=False)
        sample_features = features[indices]
        sample_labels = labels[indices]
        
        # Codifica os labels para cores
        encoded_sample_labels = self.label_encoder.transform(sample_labels)
        
        # Normaliza antes do PCA
        sample_features_scaled = self.scaler.fit_transform(sample_features)

        # PCA para visualização 2D
        pca_viz = PCA(n_components=2)
        features_2d = pca_viz.fit_transform(sample_features_scaled)
        
        # Plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                              c=encoded_sample_labels, 
                              cmap='tab10', alpha=0.7)
        plt.colorbar(scatter, label='ID do Locutor Codificado')
        plt.title('Visualização 2D das Características (PCA) por Locutor')
        plt.xlabel('Primeira Componente Principal')
        plt.ylabel('Segunda Componente Principal')
        plt.grid(True)
        plt.show()
    
    def run_pipeline(self, tar_gz_path, max_samples_initial_load=None, model_type='random_forest', split_file='validated.tsv', min_samples_per_speaker=5, max_samples_per_speaker=None):
        """
        Executa a pipeline completa para identificação de voz com Common Voice.
        
        Args:
            tar_gz_path (str): Caminho para o arquivo .tar.gz do Common Voice.
            max_samples_initial_load (int, optional): Número máximo de amostras para carregar ANTES da filtragem por locutor.
                                                    Use None para carregar todas as amostras disponíveis.
            model_type (str): Tipo de modelo para treinamento ('random_forest', 'svm').
            split_file (str): Nome do arquivo TSV a ser usado (ex: 'validated.tsv').
            min_samples_per_speaker (int): Número mínimo de amostras que um locutor deve ter para ser incluído.
            max_samples_per_speaker (int, optional): Número máximo de amostras a serem mantidas por locutor (subamostragem).
            
        Returns:
            tuple: (modelo, acurácia)
        """
        print("=== Iniciando Pipeline de Identificação de Voz com Common Voice ===\n")
        
        # 1. Extrai o dataset
        dataset_base_path = self.extract_common_voice_dataset(tar_gz_path)
        if not dataset_base_path:
            print("Falha na extração ou localização do dataset. Abortando.")
            return None, 0

        # 2. Carrega os metadados do dataset e aplica a filtragem/balanceamento
        df_metadata = self.load_common_voice_dataset(
            dataset_base_path, 
            max_samples=max_samples_initial_load, # Limite o carregamento inicial
            split_file=split_file,
            min_samples_per_speaker=min_samples_per_speaker,
            max_samples_per_speaker=max_samples_per_speaker # Passa o novo parâmetro
        )
        if df_metadata is None or df_metadata.empty:
            print("Nenhum dado válido carregado ou após filtragem. Abortando.")
            return None, 0
        
        # 3. Processa o dataset (extração de características)
        features, labels = self.process_dataset(df_metadata, dataset_base_path)
        self.features = features
        self.labels = labels

        # 4. Visualiza as características
        # print("\nVisualizando características...")
        # self.visualize_features(features, labels)
        
        # 5. Treina o modelo
        model, accuracy = self.train_model(features, labels, model_type=model_type)
        
        print(f"\n=== Pipeline Concluída ===")
        print(f"Modelo treinado com acurácia: {accuracy:.4f}")
        
        return model, accuracy

# Exemplo de uso
if __name__ == "__main__":
    # Define o caminho para o arquivo .tar.gz do Common Voice
    common_voice_tar_gz = "pt.tar" 
    
    MIN_SAMPLES_PER_SPEAKER = 80
    MAX_SAMPLES_PER_SPEAKER = 100
    
    # Inicializa a pipeline
    pipeline = VoiceIdentificationPipeline(sample_rate=16000) # Common Voice é 48kHz, mas 16kHz é comum para fala
                                                             # Se seu Common Voice pt-BR for 48kHz, considere:
                                                             # sample_rate=48000 (ou resample no preprocess_audio)
    
    # Executa a pipeline completa com filtragem de locutores
    try:
        model, accuracy = pipeline.run_pipeline(
            tar_gz_path=common_voice_tar_gz,
            max_samples_initial_load=None, # Carrega todos os metadados inicialmente (antes da filtragem)
            min_samples_per_speaker=MIN_SAMPLES_PER_SPEAKER,  # Apenas locutores com 10 ou mais amostras serão incluídos
            max_samples_per_speaker=MAX_SAMPLES_PER_SPEAKER,  # Opcional: Se um locutor tiver mais de 50 amostras, seleciona 50 aleatoriamente
            model_type='random_forest',
            split_file='validated.tsv'
        )
        
        if model is not None:
            print(f"\nModelo treinado com sucesso!")
            print(f"Acurácia final: {accuracy:.4f}")
            
            if not pipeline.features is None and len(pipeline.features) > 0:
                print("\nPreparando para exemplo de predição...")
                try:
                    temp_base_path = pipeline.extract_common_voice_dataset(common_voice_tar_gz, extract_path='cv-corpus')
                    
                    # Carrega o DF novamente com os mesmos critérios de filtragem para garantir consistência
                    temp_df = pipeline.load_common_voice_dataset(
                        temp_base_path, 
                        max_samples=None, 
                        split_file='validated.tsv',
                        min_samples_per_speaker=MIN_SAMPLES_PER_SPEAKER,
                        max_samples_per_speaker=MAX_SAMPLES_PER_SPEAKER
                    )
                    
                    if temp_df is not None and not temp_df.empty:
                        random_sample_row = temp_df.sample(1, random_state=np.random.randint(0, 1000)).iloc[0] # Amostra aleatória
                        test_audio_path = os.path.join(temp_base_path, random_sample_row['path'])
                        actual_speaker_id = random_sample_row['client_id']

                        if os.path.exists(test_audio_path):
                            print(f"\nTestando predição com amostra de áudio: {test_audio_path}")
                            print(f"ID do Locutor Real: {actual_speaker_id}")
                            
                            predicted_speaker_id, confidence = pipeline.predict_audio(test_audio_path)
                            print(f"ID do Locutor Predito: {predicted_speaker_id}")
                            print(f"Confiança da Predição: {confidence:.4f}")
                        else:
                            print(f"Aviso: Arquivo de áudio de teste não encontrado em: {test_audio_path}")
                    else:
                        print("Não foi possível carregar o dataframe temporário para amostra de teste após filtragem.")
                except Exception as e:
                    print(f"Erro ao tentar carregar amostra para predição: {e}")
            else:
                print("Não há features processadas para testar a predição.")
        
    except Exception as e:
        print(f"Erro geral na execução da pipeline: {e}")