# Movex — Como rodar treino GRU por estações (clusters) e bacias (shapefile)

Este README descreve, em português, como preparar o ambiente e como executar o pipeline existente para treinar/predizer usando o modelo GRU por estações (clusters/k-means) e por bacias (regiões do shapefile).

Baixar o dataset de: https://drive.google.com/file/d/1Ljc9hgElfpD82LmBeuKmXZjImQQ0eTnz/view?usp=drive_link

## 1) Requisitos e criação do ambiente (Conda)

- Crie um ambiente conda com Python 3.12 (nome sugerido `movex_py312`):

```bash
conda create -y -n movex python=3.12 pip
eval "$(conda shell.bash hook)"
conda activate movex
```

- Instale dependências do projeto (execute no root do repositório):

```bash
pip install -r requirements.txt
```

Observações:
- O projeto usa geopandas / shapely — em Linux é comum instalar dependências do sistema (libgdal, etc.). Se houver erro ao instalar geopandas, instale via conda:

```bash
conda install -y geopandas gdal
pip install -r requirements.txt
```

Se preferir a forma recomendada (mais confiável para dependências geoespaciais), instale via conda-forge:

```bash
# recomendado: conda-forge para geoespaciais
conda install -y -c conda-forge geopandas fiona rtree gdal pyproj shapely
pip install -r requirements.txt
```

## 2) Dados

- O loader padrão espera um arquivo processado em `data/MWC_v3_processed.parquet`.
- Se o arquivo processado não existir, o `DataLoader` pedirá na execução o caminho para o dataset bruto (parquet), então o pipeline processará e salvará em `data/MWC_v3_processed.parquet`.

Se quiser treinar apenas para um conjunto específico de estações (latitude/longitude) você pode criar um arquivo parquet filtrado e apontar o pipeline para ele (ver seção "Filtrar por estações" abaixo).

## 3) Execução interativa (modo padrão)

O projeto já fornece um entrypoint interativo: execute:

```bash
./run.sh
# ou
python -m src.main
```

Passos esperados durante a execução interativa:

1. Escolha um trimestre/estação (opções: DJF, MAM, JJA, SON) digitando 1..4.
2. O script listará combinações disponíveis de `MOVs` e `k_means` para o trimestre escolhido.
3. Digite os números dos clusters (kmeans) que deseja processar, separado por vírgula. Exemplo: `0` ou `0,7,14`.
   - O pipeline usará esses clusters para filtrar as localidades (estações) que pertencem aos pares (movs, kmeans) selecionados.
4. Na seleção de bacias/regiões (shapefile), você verá uma lista numerada das regiões hidrológicas (`RHI_NM`).
   - Para rodar em todas as bacias, digite `todos`.
   - Para selecionar apenas algumas bacias, digite os índices separados por vírgula (ex: `1,3,5`).

O pipeline então processará os anos disponíveis (o código itera por `meteorological_year` disponíveis) e para cada localidade fará o treinamento/predição. Os resultados são salvos por `save_results` implementado em `src/utils/results.py`.

## 4) Rodar apenas em bacias específicas (modo interativo)

- Quando o prompt de seleção de regiões aparecer, digite os números correspondentes às bacias desejadas (ex: `2,4`) ou `todos` para todas.
- O shapefile padrão usado pelo projeto é `SNIRH_RHI/SNIRH_RHI/SNIRH_RegioesHidrograficas.shp`.

## 5) Filtrar por estações específicas (ex.: lat/lon) — opção rápida

Se quiser executar o pipeline apenas para uma ou poucas estações específicas, você pode filtrar o arquivo processado e usar esse arquivo filtrado como entrada. Exemplo (substitua LAT/LON):

```bash
python - <<PY
import pandas as pd
df = pd.read_parquet('data/MWC_v3_processed.parquet')
df_f = df[(df.latitude==LAT) & (df.longitude==LON)]
df_f.to_parquet('data/MWC_filtered.parquet', index=False)
print('Arquivo salvo: data/MWC_filtered.parquet')
PY

# Em seguida, modifique temporariamente o caminho padrão em `src/data/loader.py` (ou renomeie) para usar 'data/MWC_filtered.parquet',
# ou substitua o arquivo processado original (faça backup antes).
```

Observação: o `DataLoader` usa por padrão `processed_path='data/MWC_v3_processed.parquet'`. Se quiser evitar editar fontes, você pode renomear/ mover seu arquivo filtrado para esse caminho (fazendo backup do original).

## 6) Parâmetros de treinamento

- O código de treinamento está em `src/models/*` e o `LocationProcessor.run_location_predictions` aceita argumentos: `epochs`, `batch_size`, `lr` e `model_type`.
- No modo interativo atual esses valores são fixos (padrões no código). Para alterá-los sem mudar o código, edite `src/processing/location.py` (linhas onde `run_location_predictions` é chamado) ou, se preferir, crie um pequeno wrapper (script) que passe parâmetros diferentes.

## 7) Saída e resultados

- Resultados de cada quadrimestre/ano são salvos pela função `save_results` definida em `src/utils/results.py`.
- Durante a execução você verá logs com: quantos grupos foram processados, quantos pontos ficaram dentro das bacias selecionadas, e possíveis erros por conjunto de features.

## 8) Problemas comuns

- Erro ao carregar shapefile: verifique se `SNIRH_RHI/SNIRH_RHI/SNIRH_RegioesHidrograficas.shp` existe e está completo (arquivos .shp, .shx, .dbf etc.).
- Dependências geoespaciais: se geopandas falhar na instalação, prefira instalar via conda (veja seção 1).
- Erros de memória/CPU: o processamento é paralelo (multiprocessing). Se a sua máquina for limitada, reduza `max_workers` no `process_map` em `src/main.py` ou execute em fewer cores.

## 9) Exemplo rápido (fluxo mínimo)

```bash
eval "$(conda shell.bash hook)"
conda activate movex
pip install -r requirements.txt
python -m src.main
# siga as interações: escolha estação (ex: 1 -> DJF), escolha clusters (ex: 0,7), escolha regiões (ex: todos)
```

## 10) Suporte / próximos passos

- Se quiser que eu adicione um modo não-interativo (CLI) que aceite flags como `--season`, `--clusters`, `--regions` e parâmetros de treino, eu posso criar esse script (ou um utilitário) para você. Diga se prefere um script novo (`scripts/run_gru.py`) ou uma flag para `src/main.py`.

---

Arquivo criado automaticamente para instruir como usar o pipeline GRU no repositório.
