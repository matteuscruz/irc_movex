# First Steps — Como rodar um treinamento (passo a passo)

Este documento mostra, em português, os passos mínimos para preparar o ambiente, instalar dependências e rodar um treinamento interativo usando o pipeline do projeto (escolher estação/trimestre, MOVs (clusters) e bacias). Siga os passos abaixo no diretório raiz do repositório.

## 1) Criar e ativar o ambiente Conda (recomendado)

Recomendado usar um ambiente dedicado. Neste guia usamos o nome `movex_py312`.

```bash
# criar ambiente com Python 3.12
conda create -y -n movex_py312 python=3.12 pip

# ativar (roda no shell bash)
eval "$(conda shell.bash hook)"
conda activate movex_py312
```

Observação: se você prefere outro nome para o ambiente, substitua `movex_py312` nos comandos abaixo.

## 2) Instalar dependências

Recomendação (mais confiável para dependências geoespaciais): instalar pacotes geoespaciais via conda-forge antes de instalar o restante via pip.

```bash
# instalar geopandas, fiona, rtree, gdal e bibliotecas geoespaciais via conda-forge
conda install -y -c conda-forge geopandas fiona rtree gdal pyproj shapely

# depois instale o restante das dependências do projeto
pip install -r requirements.txt
```

Nota sobre PyTorch (GPU):
- Se quiser suporte CUDA, instale PyTorch usando o canal oficial e escolha a versão CUDA apropriada. Exemplo (ajuste versão CUDA conforme sua GPU/driver):

```bash
# CPU-only (conda):
conda install -y pytorch torchaudio cpuonly -c pytorch

# ou com CUDA (exemplo cudatoolkit=12.1):
conda install -y pytorch torchaudio -c pytorch -c nvidia cudatoolkit=12.1
```

Se optar por instalar PyTorch via conda, instale-o antes de `pip install -r requirements.txt` para evitar conflitos.

## 3) Verificações rápidas

No terminal com o ambiente ativado, verifique as principais importações:

```bash
python -c "import torch; import geopandas as gpd; print('torch', getattr(torch,'__version__', None), 'geopandas', gpd.__version__)"
```

Se algum pacote falhar na importação, instale conforme as mensagens de erro ou use conda-forge para bibliotecas que dependem de GDAL/Fiona.

## 4) Executar o pipeline (modo interativo)

Com o ambiente ativado e dependências instaladas, rode:

```bash
python -m src.main
```

Fluxo interativo esperado (passo a passo):

1. Seleção do trimestre/estação:
   - Você verá uma lista numerada: 1. DJF, 2. MAM, 3. JJA, 4. SON
   - Digite o número correspondente (ex: `1`) e pressione Enter.

2. Seleção de clusters (MOVs/kmeans):
   - O script listará as combinações disponíveis de `MOVs` e `k_means` para a estação escolhida.
   - Você deverá digitar os números dos clusters separados por vírgula, por exemplo: `0` ou `0,7,14`.

3. Seleção do modelo (nova opção):
   - Após escolher os clusters, será exibida a lista de modelos disponíveis:
     1. lstm
     2. gru
     3. rnn
   - Você pode digitar o número (ex: `2`) ou o nome (ex: `gru`). A escolha será usada para treinar/predizer nas localidades selecionadas.

4. Seleção de bacias/regiões (shapefile):
   - O script mostrará as regiões presentes no shapefile (`SNIRH_RHI/SNIRH_RHI/SNIRH_RegioesHidrograficas.shp`).
   - Você pode digitar `todos` para processar todas as bacias, ou uma lista de índices (ex: `1,3,5`) para rodar apenas em regiões específicas.

5. O pipeline então processará os anos disponíveis e para cada localidade fará o preparo de dados, treinamento e predição usando o modelo selecionado.

## 5) Dicas para rodar em apenas algumas estações (lat/lon)

Se quiser testar em poucas estações antes de rodar tudo:

```bash
python - <<PY
import pandas as pd
df = pd.read_parquet('data/MWC_v3_processed.parquet')
# substitua LAT e LON pelos valores reais que deseja testar
df_f = df[(df.latitude==LAT) & (df.longitude==LON)]
df_f.to_parquet('data/MWC_filtered.parquet', index=False)
print('Arquivo salvo: data/MWC_filtered.parquet')
PY

# então, temporariamente, renomeie seu arquivo filtrado para o padrão do loader
mv data/MWC_v3_processed.parquet data/MWC_v3_processed.parquet.bak
mv data/MWC_filtered.parquet data/MWC_v3_processed.parquet

# rode o pipeline (ele usará apenas as estações filtradas)
python -m src.main

# depois restaure o arquivo original
mv data/MWC_v3_processed.parquet.bak data/MWC_v3_processed.parquet
```

## 6) Troubleshooting comum

- ModuleNotFoundError para `geopandas` ou `fiona`:
  - Use `conda install -c conda-forge geopandas fiona rtree gdal`.

- Erros ao abrir o shapefile:
  - Verifique se os arquivos .shp/.shx/.dbf estão no diretório `SNIRH_RHI/SNIRH_RHI/`.

- Recursos (memória/CPU):
  - O processamento paralelo usa todos os cores por padrão. Para reduzir carga, edite `src/main.py` e reduza `max_workers` em `process_map`.

## 7) Checklist rápido

- [ ] Criar ambiente Conda e ativar
- [ ] Instalar geoespaciais via conda-forge (recomendado)
- [ ] pip install -r requirements.txt
- [ ] Verificar imports (`torch`, `geopandas`)
- [ ] Rodar `python -m src.main` e seguir seleção: trimestre -> clusters -> modelo -> bacias

---

Se quiser, eu posso:
- adicionar um script não-interativo que aceite `--season`, `--clusters`, `--regions` e `--model` (posso criar `scripts/run_gru.py`), ou
- criar uma seção no `README.md` referenciando este `First_Steps.md`.
