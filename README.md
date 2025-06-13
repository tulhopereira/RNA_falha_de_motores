# 🧠 Classificação de Falhas em Motores com Redes Neurais

Projeto de manutenção preditiva com RNA para detectar falhas em motores elétricos com base em sinais de corrente, vibração e ruído.

## 🔧 Tecnologias
- Python 3.x
- Pandas, NumPy, Matplotlib
- Scikit-learn
- TensorFlow / Keras

## 📊 Dados de Entrada
A base simulada contém os seguintes atributos:
- `corrente_rms`
- `freq_vibracao`
- `ruido_std`
- `classe` (0: normal, 1: falha leve, 2: falha severa)

## 🧠 Modelo de RNA
- 2 camadas ocultas (ReLU)
- Camada de saída softmax para classificação multiclasse
- Validação com holdout 70/30
- Acurácia final: ~99%

## 📈 Resultados
Gráficos de perda e acurácia por época incluídos no código.

## 🚀 Execução
python PROJETO_RNA_FALHAS_MOTORES.py
