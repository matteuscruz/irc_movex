import geopandas as gpd
from shapely import Point, wkt


def filter_sc_pr(grouped_data):
    """Filtra grupos de coordenadas que estão dentro de SC ou PR mantendo o formato original."""
    brazil_states = gpd.read_file("gadm41_BRA_1.json")
    sc_pr = brazil_states[brazil_states["NAME_1"].isin(["SantaCatarina", "Paraná"])]

    filtered_groups = []
    for coords, df_group in grouped_data:
        lat, lon = coords
        point = Point(lon, lat)
        if sc_pr.contains(point).any():
            filtered_groups.append((coords, df_group))

    return filtered_groups


def select_regions_from_shapefile(shapefile_path):
    """
    Interface interativa para seleção de regiões de um shapefile.

    Args:
        shapefile_path: Caminho para o arquivo shapefile (.shp)

    Returns:
        Lista de nomes de regiões selecionadas ou None para todas
    """
    gdf = gpd.read_file(shapefile_path)

    # Garante que tá em lat/lon
    if gdf.crs is not None and gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    if 'RHI_NM' not in gdf.columns:
        print("AVISO: Coluna 'RHI_NM' não encontrada no shapefile.")
        return None

    regioes_disponiveis = sorted(gdf['RHI_NM'].unique().tolist())

    print("\n" + "=" * 50)
    print("SELEÇÃO DE REGIÕES HIDROGRÁFICAS")
    print("=" * 50)

    print("\nRegiões Hidrográficas disponíveis:")
    for i, regiao in enumerate(regioes_disponiveis, start=1):
        print(f"{i}. {regiao}")

    while True:
        regiao_input = input("\nDigite o(s) número(s) da(s) região(ões) (ex: 1 ou 1,3,5) ou 'todos' para todas: ")

        if regiao_input.lower() == 'todos':
            print("Selecionando todas as regiões.")
            return None

        try:
            selected_indices = [int(x.strip()) for x in regiao_input.split(',') if x.strip().isdigit()]
            if not selected_indices:
                print("Nenhum número válido informado. Tente novamente.")
                continue

            selected_regions = []
            for idx in selected_indices:
                if 1 <= idx <= len(regioes_disponiveis):
                    selected_regions.append(regioes_disponiveis[idx - 1])
                else:
                    print(f"Índice {idx} fora do range disponível (1-{len(regioes_disponiveis)})")

            if selected_regions:
                print(f"Regiões selecionadas: {selected_regions}")
                return selected_regions
            else:
                print("Nenhuma região válida selecionada. Tente novamente.")

        except ValueError:
            print("Entrada inválida. Digite números separados por vírgula ou 'todos'.")


def filter_by_shapefile(grouped_data, shapefile_path, selected_regions=None):
    """
    Filtra grupos de coordenadas que estão dentro das regiões definidas em um shapefile.

    Args:
        grouped_data: Lista de tuplas (coords, df_group) onde coords = (lat, lon)
        shapefile_path: Caminho para o arquivo shapefile (.shp)
        selected_regions: Lista de nomes de regiões hidrográficas para filtrar
                         (ex: ['URUGUAI', 'ATLÂNTICO SUDESTE'])
                         Se None, usa todas as regiões do shapefile.

    Returns:
        Lista filtrada de grupos que estão dentro das regiões selecionadas
    """
    gdf = gpd.read_file(shapefile_path)

    # Garante que tá em lat/lon
    if gdf.crs is not None and gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    # Filtra as regiões se especificado
    if selected_regions is not None and 'RHI_NM' in gdf.columns:
        gdf = gdf[gdf['RHI_NM'].isin(selected_regions)]
        print(f"Filtrando shapefile por regiões: {selected_regions}")
        print(f"Regiões encontradas após filtro: {gdf['RHI_NM'].unique().tolist()}")
    elif selected_regions is not None and 'RHI_NM' not in gdf.columns:
        print("AVISO: Coluna 'RHI_NM' não encontrada no shapefile. Usando todas as regiões.")

    if gdf.empty:
        print("AVISO: Nenhuma região encontrada após filtragem. Retornando lista vazia.")
        return []

    # Mostra estatísticas do shapefile
    print(f"\nShapefile carregado: {len(gdf)} região(ões)")
    if 'RHI_NM' in gdf.columns:
        print(f"Regiões ativas: {gdf['RHI_NM'].unique().tolist()}")

    # Processa os pontos
    filtered_groups = []
    pontos_processados = 0
    pontos_dentro = 0

    print(f"\nProcessando {len(grouped_data)} grupos de coordenadas...")

    for coords, df_group in grouped_data:
        lat, lon = coords
        point = Point(lon, lat)  # ordem correta
        pontos_processados += 1

        # Verifica se o ponto está dentro de alguma das regiões
        if gdf.covers(point).any():  # covers é mais seguro que contains
            filtered_groups.append((coords, df_group))
            pontos_dentro += 1

    print(f"Pontos processados: {pontos_processados}")
    print(f"Pontos dentro das regiões selecionadas: {pontos_dentro}")
    print(f"Taxa de retenção: {pontos_dentro / pontos_processados * 100:.1f}%")
    return filtered_groups