# CIFAR-10 - DCC197 - Visão Computacional

Um projeto de deep learning para classificação de imagens do dataset CIFAR-10 usando diferentes arquiteturas de redes neurais com PyTorch.

## Descrição

Este projeto implementa e compara diferentes modelos de visão computacional para classificação de imagens em 10 categorias usando o dataset CIFAR-10. Os modelos suportados incluem:

- **VGG16**: Rede neural customizada baseada na arquitetura VGG16
- **VGG16 Pré-treinada**: VGG16 com pesos pré-treinados do ImageNet
- **ResNet50**: Rede residual com 50 camadas (pré-treinada)
- **DenseNet121**: Rede densa com 121 camadas (pré-treinada)

## Recursos

- Suporte a múltiplas arquiteturas de redes neurais
- Dropout e Batch Normalization configuráveis
- Data augmentation para melhorar generalização
- Congelamento (freeze) de backbone com descongelamento seletivo
- Otimizador Adam com scheduler CosineAnnealing
- Visualização de resultados com Tensorboard
- Geração de matriz de confusão
- Exportação de resultados para Excel
- Execução em batch de múltiplos experimentos

## Pré-requisitos

- Python 3.7+
- GPU NVIDIA (opcional, mas recomendado)

## Instalação

1. Clone ou baixe o projeto:
```bash
clone https://github.com/gabrielctavares/DCC197-VisaoComputacional.git
cd ./DCC197-VisaoComputacional
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

As dependências incluem:
- `torch` e `torchvision` - Deep learning framework
- `numpy` e `scikit-learn` - Processamento de dados
- `matplotlib` e `seaborn` - Visualização
- `tensorboard` - Monitoramento de treinamento
- `Pillow` - Processamento de imagens
- `tqdm` - Barras de progresso

## Estrutura do Projeto

```
.
├── experiments/                   # Configurações de experimentos
│   ├── experiments_part1.json
│   └── experiments_part2.json
├── results/                       # Resultados dos treinamentos
├── src/
│   ├── models.py                 # Definição dos modelos
│   ├── train.py                  # Script principal de treinamento
│   └── log_util.py               # Utilidades de logging
├── run.py                         # Executor de experimentos em batch
├── run.sh                         # Script shell para executar
└── requirements.txt               # Dependências Python
```

## Como Usar

### 1. Treinar um modelo único

Use o script `train.py` para treinar um modelo com configurações específicas:

```bash
python src/train.py \
    --id exp_1 \
    --model_name vgg16 \
    --epochs 20 \
    --batch_size 64 \
    --img_size 32 \
    --learning_rate 0.001 \
    --weight_decay 0.0005
```

#### Argumentos disponíveis:

| Argumento | Tipo | Padrão | Descrição |
|-----------|------|--------|-----------|
| `--id` | str | Obrigatório | Identificador do experimento |
| `--model_name` | str | Obrigatório | Nome do modelo (vgg16, vgg16_pretreinada, resnet50, densenet121) |
| `--epochs` | int | 100 | Número de épocas |
| `--batch_size` | int | 32 | Tamanho do batch |
| `--img_size` | int | 64 | Tamanho das imagens |
| `--learning_rate` | float | 1e-4 (0.0001) | Taxa de aprendizado |
| `--weight_decay` | float | 1e-4 (0.0001) | L2 regularization |
| `--use_dropout` | bool | false | Usar dropout |
| `--dropout_rate` | float | 0.5 | Taxa de dropout |
| `--use_batch_norm` | bool | false | Usar Batch Normalization |
| `--use_data_augmentation` | bool | false | Aplicar data augmentation |
| `--freeze_backbone` | bool | false | Congelar pesos da backbone |
| `--unfreeze_last_n_params` | int | 0 | Número de parâmetros para descongelar |
| `--results_file` | str | resultados_YYYYMMDD.xlsx | Arquivo para salvar resultados |

### 2. Executar múltiplos experimentos em batch

Os experimentos são definidos em arquivos JSON na pasta `experiments/`:

```bash
python run.py
```

Ou para executar um experimento específico:

```bash
python run.py --id vgg16_baseline
```

#### Formato do arquivo de experimento (JSON):

```json
[
  {
    "id": "meu_experimento",
    "model_name": "vgg16",
    "epochs": 20,
    "img_size": 32,
    "batch_size": 64,
    "learning_rate": 0.001,
    "weight_decay": 0.0005,
    "use_dropout": true,
    "dropout_rate": 0.5,
    "use_batch_norm": true,
    "use_data_augmentation": false,
    "freeze_backbone": false,
    "unfreeze_last_n_params": 0
  }
]
```

### 3. Monitorar treinamento com Tensorboard

Durante o treinamento, você pode visualizar métricas em tempo real:

```bash
tensorboard --logdir=./runs
```

Então acesse `http://localhost:6006` no navegador.

## Dataset

O projeto usa o dataset **CIFAR-10**, que contém:
- 60.000 imagens coloridas (32x32 pixels)
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 50.000 imagens de treinamento
- 10.000 imagens de teste


## Exemplos de Uso

### Exemplo 1: Treinar VGG16 simples
```bash
python src/train.py \
    --id vgg16_test \
    --model_name vgg16 \
    --epochs 5 \
    --batch_size 64 \
    --img_size 32 \
    --learning_rate 0.001 \
    --weight_decay 0
```

### Exemplo 2: Treinar ResNet50 pré-treinada com freeze
```bash
python src/train.py \
    --id resnet50_transfer \
    --model_name resnet50 \
    --epochs 10 \
    --batch_size 32 \
    --img_size 224 \
    --learning_rate 0.0001 \
    --weight_decay 0.0005 \
    --freeze_backbone True \
    --unfreeze_last_n_params 10
```

### Exemplo 3: Executar experimentos em batch
```bash
python run.py
```

## Saída e Resultados

Após o treinamento, você encontrará:

- **Arquivo Excel** (`results_*.xlsx`): Métricas detalhadas por classe e accuracy geral
- **Logs Tensorboard** (`runs/`): Gráficos de loss e acurácia
- **Matriz de Confusão**: Visualização do desempenho por classe
- **Log do console**: Progresso detalhado do treinamento

## Arquitetura dos Modelos

### VGG16 (Customizado)
Rede convolucional clássica com dropout e batch normalization opcionais.

### VGG16 Pré-treinada
VGG16 com pesos do ImageNet, com último layer ajustado para 10 classes do CIFAR-10.

### ResNet50 (Transfer Learning)
ResNet50 pré-treinada com possibilidade de congelamento da backbone e fine-tuning.

### DenseNet121 (Transfer Learning)
DenseNet121 pré-treinada com dense connections para melhor fluxo de gradientes.

## Configuração de GPU

O projeto detecta automaticamente a disponibilidade de GPU:
- Se CUDA está disponível, usa GPU
- Caso contrário, usa CPU

## Logs e Debugging

Os logs detalhados são salvos em `run_outputs/` com timestamp. 
