# 🧠 Exercise Pose Trainers

Este repositório contém os scripts de treinamento de modelos de machine learning para validação da execução do exercício prancha alta (high plank), utilizando dados de ângulos extraídos por pose estimation com o BlazePose (MediaPipe).

## 📁 Estrutura do Repositório

- A pasta `high_plank_multimodel` contém as imagens para treinamento e seus rótulos em `labels.json`.
- A pasta `test` contém imagens que podem ser usadas para teste dos modelos treinados através do comando `poetry run test`.
- As outras pastas possuem os códigos para treinamento dos respectivos modelos.

## ⚙️ Comandos

É neceesário ter o [Poetry](https://python-poetry.org/) instalado para rodar os comandos a seguir.

### Instalar dependências

Navegue até o diretório do modelo (por exemplo, KNN):
```
cd exercise-pose-trainer-knn-angles-multimodel
poetry install
```

### Treinar modelo

Treina o modelo e o exporta como arquivo `.pkl`.

```
poetry run train [--seed SEED] [--plot] path
```

- `seed`: seed usada em `random_state` de `train_test_split`. Útil para reprodutibilidade.
- `plot`: Deve ou não plotar gráficos do modelo ao fim do treinamento.
- `path`: Caminho para pasta contendo imagens e labels. A pasta deve conter pasta `images` com as imagens e arquivo `labels.json`, a exemplo da pasta `high_plank_multimodel`.

### Ver report

Plota gráficos e printa métricas do modelo.

```
poetry run report model_path
```

- `model_path`: Caminho para o modelo `.pkl` obtido do treinamento.

### Testar modelo

Testa modelo treinado em imagens contidas em uma pasta.

```
poetry run test test_path model_path
```

- `test_path`: Caminho da pasta contendo imagens a serem testadas.
- `model_path`: Caminho para o modelo `.pkl` obtido do treinamento.

### Exportar ONNX

Converte arquivo `.pkl` obtido do treinamento em arquivo `.onnx`.

```
poetry run export_onnx model_path
```

- `model_path`: Caminho para o modelo `.pkl` obtido do treinamento.
