#!/usr/bin/env python3
"""
Voice Recognition System Using FFT - TIMIT Dataset Pipeline
Sistema de Reconhecimento de Voz usando FFT - Pipeline com Dataset TIMIT

Implementa uma pipeline completa para reconhecimento de voz utilizando:
- Dataset TIMIT via Deeplake
- Extração de características usando FFT
- Classificação com Machine Learning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
import deeplake
import warnings
warnings.filterwarnings('ignore')

class VoiceIdentificationPipeline:
    """Pipeline principal para reconhecimento de voz com dataset TIMIT"""
    
    def __init__(self, sample_rate=16000, n_fft=2048, hop_length=512):
        """
        Inicializa a pipeline de reconhecimento de voz
        
        Args:
            sample_rate (int): Taxa de amostragem do áudio
            n_fft (int): Tamanho da janela FFT
            hop_length (int): Tamanho do passo para análise
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.features = None
        self.labels = None
        
    def load_timit_dataset(self, max_samples=1000):
        """
        Carrega o dataset TIMIT via Deeplake
        
        Args:
            max_samples (int): Número máximo de amostras para carregar
            
        Returns:
            deeplake.Dataset: Dataset carregado
        """
        print("Carregando dataset TIMIT via Deeplake...")
        try:
            # Carrega o dataset TIMIT de treinamento
            ds = deeplake.load("hub://activeloop/timit-train")
            print(f"Dataset carregado com sucesso!")
            print(f"Número total de amostras: {len(ds)}")
            
            # Limita o número de amostras se especificado
            if max_samples and len(ds) > max_samples:
                print(f"Limitando a {max_samples} amostras para processamento")
                
            return ds[:max_samples] 
            
        except Exception as e:
            print(f"Erro ao carregar dataset: {e}")
            return None
    
    def preprocess_audio(self, audio_data, target_length=None):
        """
        Pré-processa os dados de áudio
        
        Args:
            audio_data (np.array): Dados de áudio
            target_length (int): Comprimento alvo para normalização
            
        Returns:
            np.array: Áudio pré-processado
        """
        # Normaliza o áudio
        audio_data = audio_data.astype(np.float32)
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Aplica filtro passa-alta para remover ruído de baixa frequência
        sos = signal.butter(5, 80, btype='high', fs=self.sample_rate, output='sos')
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
        Extrai características usando FFT
        
        Args:
            audio_data (np.array): Dados de áudio
            
        Returns:
            np.array: Características extraídas
        """
        features = []
        
        # 1. FFT Spectral Features
        fft_spectrum = np.abs(fft(audio_data))
        fft_spectrum = fft_spectrum[:len(fft_spectrum)//2]  # Apenas metade positiva
        
        # Características espectrais básicas
        spectral_centroid = np.sum(fft_spectrum * np.arange(len(fft_spectrum))) / np.sum(fft_spectrum)
        spectral_rolloff = np.percentile(fft_spectrum, 85)
        spectral_bandwidth = np.sqrt(np.sum(((np.arange(len(fft_spectrum)) - spectral_centroid) ** 2) * fft_spectrum) / np.sum(fft_spectrum))
        
        features.extend([spectral_centroid, spectral_rolloff, spectral_bandwidth])
        
        # 2. Características estatísticas do espectro
        features.extend([
            np.mean(fft_spectrum),
            np.std(fft_spectrum),
            np.max(fft_spectrum),
            np.min(fft_spectrum),
            np.median(fft_spectrum)
        ])
        
        # 3. Energia em bandas de frequência específicas
        freqs = fftfreq(len(audio_data), 1/self.sample_rate)[:len(fft_spectrum)]
        
        # Bandas de frequência para análise da fala
        bands = [(0, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
        for low, high in bands:
            band_mask = (freqs >= low) & (freqs <= high)
            band_energy = np.sum(fft_spectrum[band_mask])
            features.append(band_energy)
        
        # 4. MFCC usando librosa (complementa FFT)
        try:
            mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
            mfcc_features = [
                np.mean(mfccs, axis=0),
                np.std(mfccs, axis=0),
                np.max(mfccs, axis=0),
                np.min(mfccs, axis=0)
            ]
            features.extend(np.concatenate(mfcc_features).flatten())
        except:
            # Se falhar, adiciona zeros
            features.extend([0] * 52)  # 13 MFCCs * 4 estatísticas
        
        # 5. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        features.extend([np.mean(zcr), np.std(zcr)])
        
        return np.array(features)
    
    def extract_phoneme_label(self, phoneme_data):
        """
        Extrai e processa labels de fonemas do dataset TIMIT
        
        Args:
            phoneme_data: Dados de fonemas do dataset
            
        Returns:
            str: Fonema principal ou categoria
        """
        if phoneme_data is None or len(phoneme_data) == 0:
            return "unknown"
        
        # Extrai o fonema mais frequente ou o primeiro
        if hasattr(phoneme_data, 'data'):
            phonemes = phoneme_data.data
        else:
            phonemes = phoneme_data
            
        if isinstance(phonemes, (list, np.ndarray)) and len(phonemes) > 0:
            # Retorna o primeiro fonema se for uma lista
            return str(phonemes[0]) if phonemes[0] is not None else "unknown"
        
        return str(phonemes) if phonemes is not None else "unknown"
    
    def process_dataset(self, dataset, max_samples=1000):
        """
        Processa todo o dataset extraindo características e labels
        
        Args:
            dataset: Dataset TIMIT carregado
            max_samples (int): Número máximo de amostras para processar
            
        Returns:
            tuple: (features, labels)
        """
        print("Processando dataset e extraindo características...")
        
        features_list = []
        labels_list = []
        processed_count = 0
        
        # Determina o número de amostras para processar
        total_samples = min(len(dataset), max_samples) if max_samples else len(dataset)
        
        for i in range(total_samples):
            try:
                # Carrega dados de áudio
                audio_data = np.array(dataset[i].audios.data()['value'])
                
                # Verifica se os dados de áudio são válidos
                if len(audio_data) == 0:
                    continue
                
                # Pré-processa o áudio
                processed_audio = self.preprocess_audio(audio_data)
                
                # Extrai características
                features = self.extract_fft_features(processed_audio)
                
                # Extrai label (fonema ou texto)
                try:
                    # Tenta extrair fonema
                    if hasattr(dataset[i], 'speaker_ids'):
                        label = str(dataset[i].speaker_ids.data()['value'][0]) if dataset[i].speaker_ids.data()['value'][0] else 0
                    else:
                        label = 0  # Label genérico
                except:
                    label = 0
                
                features_list.append(features)
                labels_list.append(label)
                processed_count += 1
                
                # Progresso
                if processed_count % 100 == 0:
                    print(f"Processadas {processed_count}/{total_samples} amostras...")
                    
            except Exception as e:
                print(f"Erro ao processar amostra {i}: {e}")
                continue
        
        print(f"Processamento concluído: {processed_count} amostras válidas")
        
        if len(features_list) == 0:
            raise ValueError("Nenhuma amostra válida foi processada")
        
        # Converte para arrays numpy
        features_array = features_list
        labels_array = labels_list
        
        return features_array, labels_array
    
    def train_model(self, features, labels, test_size=0.2, model_type='random_forest'):
        """
        Treina o modelo de classificação
        
        Args:
            features (np.array): Características extraídas
            labels (np.array): Labels correspondentes
            test_size (float): Proporção dos dados para teste
            model_type (str): Tipo de modelo ('random_forest', 'svm')
            
        Returns:
            tuple: (modelo_treinado, acurácia_teste)
        """
        print(f"Treinando modelo {model_type}...")
        
        # Codifica os labels
        encoded_labels = self.label_encoder.fit_transform(labels)
        
        # Divisão treino/teste
        X_train, X_test, y_train, y_test = train_test_split(
            features, encoded_labels, test_size=test_size, random_state=42
        )
        
        # Normalização das características
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Redução de dimensionalidade se necessário
        # if X_train_scaled.shape[1] > 50:
        #     pca = PCA(n_components=50)
        #     X_train_scaled = pca.fit_transform(X_train_scaled)
        #     X_test_scaled = pca.transform(X_test_scaled)
        #     self.pca = pca
        
        # Escolha do modelo
        if model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_type == 'svm':
            model = SVC(kernel='rbf', random_state=42, probability=True)
        else:
            raise ValueError("Tipo de modelo não suportado")
        
        # Treinamento
        model.fit(X_train_scaled, y_train)
        
        # Avaliação
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Acurácia do modelo: {accuracy:.4f}")
        
        # Relatório detalhado
        print("\nRelatório de Classificação:")
        print(classification_report(y_test, y_pred#, target_names=self.label_encoder.classes_[:len(np.unique(y_test))]
                                    ))
        
        self.model = model
        return model, accuracy
    
    def predict_audio(self, audio_data):
        """
        Prediz a classe de um áudio
        
        Args:
            audio_data (np.array): Dados de áudio
            
        Returns:
            tuple: (classe_predita, probabilidade)
        """
        if self.model is None:
            raise ValueError("Modelo não foi treinado ainda")
        
        # Pré-processa o áudio
        processed_audio = self.preprocess_audio(audio_data)
        
        # Extrai características
        features = self.extract_fft_features(processed_audio)
        features = features.reshape(1, -1)
        
        # Normaliza
        features_scaled = self.scaler.transform(features)
        
        # Aplica PCA se foi usado no treinamento
        if hasattr(self, 'pca'):
            features_scaled = self.pca.transform(features_scaled)
        
        # Predição
        prediction = self.model.predict(features_scaled)[0]
        probability = np.max(self.model.predict_proba(features_scaled))
        
        # Decodifica o label
        predicted_label = self.label_encoder.inverse_transform([prediction])[0]
        
        return predicted_label, probability
    
    def visualize_features(self, features, labels, n_samples=100):
        """
        Visualiza as características extraídas
        
        Args:
            features (np.array): Características
            labels (np.array): Labels
            n_samples (int): Número de amostras para visualizar
        """
        # Seleciona uma amostra aleatória
        indices = np.random.choice(len(features), min(n_samples, len(features)), replace=False)
        sample_features = features[indices]
        sample_labels = labels[indices]
        
        # PCA para visualização 2D
        pca_viz = PCA(n_components=2)
        features_2d = pca_viz.fit_transform(sample_features)
        
        # Plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                             c=self.label_encoder.fit_transform(sample_labels), 
                             cmap='tab10', alpha=0.7)
        plt.colorbar(scatter)
        plt.title('Visualização 2D das Características (PCA)')
        plt.xlabel('Primeira Componente Principal')
        plt.ylabel('Segunda Componente Principal')
        plt.show()
    
    def run_pipeline(self, max_samples=1000, model_type='random_forest'):
        """
        Executa a pipeline completa
        
        Args:
            max_samples (int): Número máximo de amostras para processar
            model_type (str): Tipo de modelo para treinamento
            
        Returns:
            tuple: (modelo, acurácia)
        """
        print("=== Iniciando Pipeline de Reconhecimento de Voz ===\n")
        
        # 1. Carrega o dataset
        dataset = self.load_timit_dataset(max_samples)
        if dataset is None:
            return None, 0
        
        # 2. Processa o dataset
        features, labels = self.process_dataset(dataset, max_samples)
        self.features = features
        self.labels = labels
        
        print(f"\nForma das características: {features.shape}")
        print(f"Número de classes únicas: {len(np.unique(labels))}")
        print(f"Classes: {np.unique(labels)[:10]}...")  # Mostra primeiras 10 classes
        
        # 3. Visualiza as características
        print("\nVisualizando características...")
        self.visualize_features(features, labels)
        
        # 4. Treina o modelo
        model, accuracy = self.train_model(features, labels, model_type=model_type)
        
        print(f"\n=== Pipeline Concluída ===")
        print(f"Modelo treinado com acurácia: {accuracy:.4f}")
        
        return model, accuracy

# Exemplo de uso
if __name__ == "__main__":
    # Inicializa a pipeline
    pipeline = VoiceIdentificationPipeline()
    
    # Executa a pipeline completa
    try:
        model, accuracy = pipeline.run_pipeline(max_samples=10, model_type='random_forest')
        
        if model is not None:
            print(f"\nModelo treinado com sucesso!")
            print(f"Acurácia final: {accuracy:.4f}")
            
            # Exemplo de predição (se houver dados disponíveis)
            if pipeline.features is not None and len(pipeline.features) > 0:
                # Testa com uma amostra aleatória
                sample_idx = np.random.randint(0, len(pipeline.features))
                sample_audio = pipeline.features[sample_idx]  # Simula áudio com características
                
                print(f"\nTestando predição com amostra {sample_idx}...")
                print(f"Label real: {pipeline.labels[sample_idx]}")
                
                # Nota: Para predição real, precisaríamos dos dados de áudio originais
                # Aqui é apenas um exemplo da estrutura
        
    except Exception as e:
        print(f"Erro na execução da pipeline: {e}")
        print("Verifique se o dataset TIMIT está acessível via Deeplake")