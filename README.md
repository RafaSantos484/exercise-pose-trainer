# 🧠 Exercise Pose Trainers

Este repositório contém os scripts de treinamento de modelos de machine learning para validação da execução do exercício prancha alta (high plank), utilizando dados de ângulos extraídos por pose estimation com o BlazePose (MediaPipe).

## 📁 Estrutura do Repositório

- A pasta `high_plank_imgs` contém as imagens para treinamento separados por pastas, onde o nome de cada pasta representa o rótulo de suas imagens.
- A pasta `test` contém imagens que podem ser usadas para teste dos modelos treinados através do comando `poetry run test`.

## ⚙️ Comandos

É neceesário ter o [Poetry](https://python-poetry.org/) instalado para rodar os comandos a seguir.

### Instalar dependências

```
poetry install
```

### Treinar modelo

Treina o modelo e o exporta como arquivo `.pkl`.

```
poetry run train [--seed SEED] path {fcnn,gradient_boosting,logistic_regression,random_forest,svm}
```

- `seed`: seed usada em `random_state` de `train_test_split`. Útil para reprodutibilidade.
- `path`: Caminho para pasta contendo pastas com nome das classes e suas respectivas imagens, a exemplo da pasta `high_plank_imgs`.

### Ver report

Plota gráficos e printa métricas do modelo.

```
poetry run view_report {fcnn,gradient_boosting,logistic_regression,random_forest,svm}
```

### Testar modelo

Testa modelo treinado em imagens contidas em uma pasta.

```
poetry run test test_path {fcnn,gradient_boosting,logistic_regression,random_forest,svm}
```

- `test_path`: Caminho da pasta contendo imagens a serem testadas.
