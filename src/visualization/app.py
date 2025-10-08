"""
Главное приложение для интерактивной 3D визуализации иерархической кластеризации.
"""

import json
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import os
import sys

# Добавляем путь к корню проекта
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

# Загружаем конфигурацию
with open('../../config.json', 'r') as f:
    config = json.load(f)

# Параметры визуализации
SPHERE_SCALE = config['visualization']['sphere_scale_factor']
PORT = config['visualization']['port']
SLIDER_STEPS = config['visualization']['slider_steps']

# Загружаем кэш кластеризации
print("Загрузка кэша кластеризации...")
with open('../../data/clustering_cache.json', 'r', encoding='utf-8') as f:
    clustering_cache = json.load(f)

# Получаем доступные уровни кластеризации
available_levels = sorted([int(k) for k in clustering_cache.keys()])
min_clusters = min(available_levels)
max_clusters = max(available_levels)

print(f"✓ Загружен кэш для {len(available_levels)} уровней")
print(f"  Диапазон: {min_clusters} - {max_clusters} кластеров")


def slider_value_to_n_clusters(slider_value: float) -> int:
    """
    Конвертирует значение слайдера (0-100) в количество кластеров (логарифмическая шкала).

    Args:
        slider_value: Значение слайдера (0-100)

    Returns:
        Количество кластеров
    """
    if slider_value == 0:
        return 1
    elif slider_value == 100:
        return max_clusters

    # Логарифмическая интерполяция
    log_min = np.log(1)
    log_max = np.log(max_clusters)
    log_value = log_min + (log_max - log_min) * (slider_value / 100)
    n_clusters = int(np.exp(log_value))

    # Находим ближайший доступный уровень
    n_clusters = min(available_levels, key=lambda x: abs(x - n_clusters))
    return n_clusters


def n_clusters_to_slider_value(n_clusters: int) -> float:
    """
    Конвертирует количество кластеров обратно в значение слайдера.

    Args:
        n_clusters: Количество кластеров

    Returns:
        Значение слайдера (0-100)
    """
    if n_clusters <= 1:
        return 0
    elif n_clusters >= max_clusters:
        return 100

    log_min = np.log(1)
    log_max = np.log(max_clusters)
    log_value = np.log(n_clusters)

    slider_value = ((log_value - log_min) / (log_max - log_min)) * 100
    return slider_value


def calculate_sphere_size(doc_count: int, scale_factor: float, formula: str) -> float:
    """
    Вычисляет размер сферы на основе количества документов.

    Args:
        doc_count: Количество документов в кластере
        scale_factor: Коэффициент масштабирования
        formula: Название формулы ('formula1', 'formula2', 'formula3')

    Returns:
        Размер сферы
    """
    if doc_count == 0:
        return 1.0

    n = doc_count

    if formula == 'formula1':
        # sqrt(n) * log(n) / scale
        return (np.sqrt(n) * np.log(n + 1)) / scale_factor
    elif formula == 'formula2':
        # n^(1/3) / scale (кубический корень)
        return (n ** (1/3)) / scale_factor
    elif formula == 'formula3':
        # log(n) / scale (только логарифм)
        return np.log(n + 1) / scale_factor
    else:
        # По умолчанию formula1
        return (np.sqrt(n) * np.log(n + 1)) / scale_factor


def create_3d_scatter(n_clusters: int, scale_factor: float, formula: str):
    """
    Создаёт 3D scatter plot для заданного количества кластеров.

    Args:
        n_clusters: Количество кластеров
        scale_factor: Коэффициент масштабирования сфер
        formula: Формула вычисления размера сфер

    Returns:
        Plotly Figure
    """
    # Находим ближайший уровень в кэше
    n_clusters = min(available_levels, key=lambda x: abs(x - n_clusters))

    level_data = clustering_cache[str(n_clusters)]
    clusters = level_data['clusters']

    # Используем t-SNE для 3D визуализации центроидов
    from sklearn.manifold import TSNE

    # Собираем центроиды
    centroids = np.array([cluster['centroid'] for cluster in clusters])

    # Снижаем размерность до 3D
    if len(centroids) > 2:
        # Используем t-SNE для лучшей визуализации
        perplexity = min(30, len(centroids) - 1)
        tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity)
        coords_3d = tsne.fit_transform(centroids)
    elif len(centroids) == 2:
        # Если два кластера, размещаем по оси X
        coords_3d = np.array([[-1, 0, 0], [1, 0, 0]])
    else:
        # Если только один кластер, размещаем в центре
        coords_3d = np.array([[0, 0, 0]])

    # Создаём traces для каждого кластера
    traces = []

    for i, cluster in enumerate(clusters):
        x, y, z = coords_3d[i]
        size = calculate_sphere_size(cluster['doc_count'], scale_factor, formula)

        # Размер маркера (масштабируем для видимости)
        marker_size = size * 100

        trace = go.Scatter3d(
            x=[x],
            y=[y],
            z=[z],
            mode='markers+text',
            name=cluster['name'],
            text=[cluster['name']],
            textposition='top center',
            marker=dict(
                size=marker_size,
                color=cluster['color'],
                line=dict(color='white', width=2),
                opacity=0.8
            ),
            customdata=[[
                cluster['cluster_id'],
                cluster['name'],
                cluster['doc_count'],
                cluster['color']
            ]],
            hovertemplate=(
                '<b>%{text}</b><br>'
                'Кластер ID: %{customdata[0]}<br>'
                'Документов: %{customdata[2]}<br>'
                '<extra></extra>'
            )
        )
        traces.append(trace)

    fig = go.Figure(data=traces)

    fig.update_layout(
        title=f'3D визуализация кластеров ({n_clusters} кластеров)',
        scene=dict(
            xaxis=dict(title='X', showgrid=True, showbackground=True),
            yaxis=dict(title='Y', showgrid=True, showbackground=True),
            zaxis=dict(title='Z', showgrid=True, showbackground=True),
            bgcolor='rgb(240, 240, 240)'
        ),
        showlegend=False,
        hovermode='closest',
        height=800,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    return fig, n_clusters  # Возвращаем также реальное количество кластеров


def create_cluster_list_items(n_clusters: int):
    """
    Создаёт список элементов кластеров для правой панели.

    Args:
        n_clusters: Количество кластеров

    Returns:
        Список HTML элементов
    """
    # Находим ближайший уровень в кэше
    n_clusters = min(available_levels, key=lambda x: abs(x - n_clusters))

    level_data = clustering_cache[str(n_clusters)]
    clusters = level_data['clusters']

    items = []
    for cluster in clusters:
        item = dbc.ListGroupItem([
            html.Div([
                html.Div(
                    style={
                        'width': '15px',
                        'height': '15px',
                        'backgroundColor': cluster['color'],
                        'borderRadius': '50%',
                        'display': 'inline-block',
                        'marginRight': '8px',
                        'border': '1px solid white',
                        'flexShrink': '0'
                    }
                ),
                html.Span(
                    f"#{cluster['cluster_id']}: ",
                    style={'fontWeight': 'bold', 'fontSize': '0.8rem'}
                ),
                html.Span(cluster['name'], style={'fontSize': '0.8rem'})
            ], style={'display': 'flex', 'alignItems': 'center'})
        ], id={'type': 'cluster-item', 'index': cluster['cluster_id']}, style={'padding': '0.4rem 0.6rem'})
        items.append(item)

    return items


# Инициализация Dash приложения
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout приложения
app.layout = dbc.Container([
    dbc.Row([
        # Левая панель (25%) - список кластеров
        dbc.Col([
            html.Div([
                html.H6("Список кластеров", className="mb-2", style={'fontSize': '0.95rem'}),
                dbc.ListGroup(
                    id='cluster-list',
                    children=create_cluster_list_items(max_clusters // 10),
                    style={
                        'maxHeight': '95vh',
                        'overflowY': 'auto',
                        'fontSize': '0.85rem'
                    }
                )
            ], style={'padding': '10px', 'height': '100vh', 'overflowY': 'auto'})
        ], width=3),

        # Центральная панель (50%) - 3D визуализация
        dbc.Col([
            dcc.Graph(
                id='3d-scatter',
                figure=create_3d_scatter(max_clusters // 10, SPHERE_SCALE, 'formula1')[0],  # Берём только figure
                style={'height': '100vh'}
            )
        ], width=6),

        # Правая панель (25%) - управление и настройки
        dbc.Col([
            html.Div([
                # Заголовок
                html.H6("Настройки", className="mb-3", style={'fontSize': '0.95rem'}),

                # Слайдер количества кластеров
                html.Label("Количество кластеров:", className="mb-1", style={'fontSize': '0.85rem'}),
                dcc.Slider(
                    id='cluster-slider',
                    min=0,
                    max=100,
                    step=1,
                    value=n_clusters_to_slider_value(max_clusters // 10),
                    marks={},  # Убираем метки
                    tooltip={"placement": "bottom", "always_visible": True}
                ),

                # Индикатор количества кластеров
                html.Div([
                    html.P(id='cluster-count-display', className="text-center mt-1 mb-2", style={'fontSize': '0.85rem', 'fontWeight': 'bold'})
                ]),

                html.Hr(style={'margin': '0.5rem 0'}),

                # Слайдер масштаба сфер
                html.Label("Масштаб сфер:", className="mb-1", style={'fontSize': '0.85rem'}),
                dcc.Slider(
                    id='scale-slider',
                    min=10,
                    max=200,
                    step=5,
                    value=SPHERE_SCALE,
                    marks={},  # Убираем метки
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                html.Div([
                    html.P(id='scale-display', className="text-center mt-1 mb-2 text-muted", style={'fontSize': '0.75rem'})
                ]),

                html.Hr(style={'margin': '0.5rem 0'}),

                # Выбор формулы
                html.Label("Формула радиуса:", className="mb-1", style={'fontSize': '0.85rem'}),
                dcc.Dropdown(
                    id='formula-dropdown',
                    options=[
                        {'label': 'Формула 1: √n × log(n)', 'value': 'formula1'},
                        {'label': 'Формула 2: ∛n', 'value': 'formula2'},
                        {'label': 'Формула 3: log(n)', 'value': 'formula3'}
                    ],
                    value='formula1',
                    clearable=False,
                    className="mb-2",
                    style={'fontSize': '0.8rem'}
                ),

                html.Hr(style={'margin': '0.5rem 0'}),

                # Информация о выбранном кластере
                html.Div([
                    html.H6("Выбранный кластер:", className="mb-2", style={'fontSize': '0.9rem'}),
                    html.Div(id='selected-cluster-info', children=[
                        html.P("Кликните на кластер", className="text-muted", style={'fontSize': '0.8rem'})
                    ])
                ])
            ], style={'padding': '10px', 'height': '100vh', 'overflowY': 'auto', 'fontSize': '0.85rem'})
        ], width=3)
    ], style={'margin': '0', 'padding': '0'})
], fluid=True, style={'padding': '0'})


@app.callback(
    [Output('3d-scatter', 'figure'),
     Output('cluster-list', 'children'),
     Output('cluster-count-display', 'children'),
     Output('scale-display', 'children')],
    [Input('cluster-slider', 'value'),
     Input('scale-slider', 'value'),
     Input('formula-dropdown', 'value')]
)
def update_visualization(slider_value, scale_factor, formula):
    """
    Обновляет визуализацию при изменении слайдеров и настроек.

    Args:
        slider_value: Значение слайдера количества кластеров
        scale_factor: Значение масштаба сфер
        formula: Выбранная формула

    Returns:
        Кортеж (figure, cluster_list, count_display, scale_display)
    """
    requested_n_clusters = slider_value_to_n_clusters(slider_value)
    fig, actual_n_clusters = create_3d_scatter(requested_n_clusters, scale_factor, formula)
    cluster_list = create_cluster_list_items(actual_n_clusters)
    count_display = f"Кластеров: {actual_n_clusters}"
    scale_display = f"Текущий масштаб: {scale_factor}"
    return fig, cluster_list, count_display, scale_display


@app.callback(
    Output('selected-cluster-info', 'children'),
    [Input('3d-scatter', 'clickData')],
    [State('cluster-slider', 'value')]
)
def display_click_data(clickData, slider_value):
    """
    Отображает информацию о кликнутом кластере.

    Args:
        clickData: Данные клика
        slider_value: Текущее значение слайдера

    Returns:
        HTML элементы с информацией о кластере
    """
    if clickData is None:
        return html.P("Кликните на кластер для просмотра информации", className="text-muted")

    # Извлекаем данные кластера
    point = clickData['points'][0]
    customdata = point['customdata']
    cluster_id, name, doc_count, color = customdata

    n_clusters = slider_value_to_n_clusters(slider_value)
    n_clusters = min(available_levels, key=lambda x: abs(x - n_clusters))

    # Находим полную информацию о кластере
    level_data = clustering_cache[str(n_clusters)]
    cluster_info = next(
        (c for c in level_data['clusters'] if c['cluster_id'] == cluster_id),
        None
    )

    if cluster_info is None:
        return html.P("Информация о кластере не найдена", className="text-danger")

    # Формируем отображение
    return html.Div([
        html.Div([
            html.Div(
                style={
                    'width': '20px',
                    'height': '20px',
                    'backgroundColor': color,
                    'borderRadius': '50%',
                    'display': 'inline-block',
                    'marginRight': '10px',
                    'border': '2px solid white',
                    'boxShadow': '0 1px 3px rgba(0,0,0,0.2)'
                }
            ),
            html.Strong(f"Кластер #{cluster_id}", style={'display': 'inline-block', 'verticalAlign': 'middle', 'fontSize': '0.85rem'})
        ], style={'marginBottom': '10px'}),

        html.P([
            html.Strong("Название: ", style={'fontSize': '0.8rem'}),
            html.Span(name, style={'fontSize': '0.8rem'})
        ], style={'margin': '5px 0'}),

        html.P([
            html.Strong("Документов: ", style={'fontSize': '0.8rem'}),
            html.Span(str(doc_count), style={'fontSize': '0.8rem'})
        ], style={'margin': '5px 0'}),

        html.P([
            html.Strong("Топ-5 тегов:", style={'fontSize': '0.8rem'})
        ], style={'margin': '5px 0 3px 0'}),

        html.Ul([
            html.Li(f"{tag} ({weight:.2f})", style={'fontSize': '0.75rem'})
            for tag, weight in cluster_info['top_tags'][:5]
        ], style={'marginLeft': '15px', 'paddingLeft': '5px'})
    ])


def main():
    """Запускает приложение."""
    print("\n" + "="*60)
    print("ЗАПУСК ИНТЕРАКТИВНОЙ ВИЗУАЛИЗАЦИИ")
    print("="*60)
    print(f"\nОткройте браузер: http://localhost:{PORT}")
    print(f"Доступно кластеров: {min_clusters} - {max_clusters}")
    print("\nДля остановки нажмите Ctrl+C")
    print("="*60 + "\n")

    app.run(debug=True, port=PORT, host='0.0.0.0')


if __name__ == "__main__":
    main()
