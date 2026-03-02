"""
СПРИНТ 4: MVP - Единое приложение
Объединяет генератор, модель и интерфейс
Симуляция: цена -> продажи -> новая цена
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================
# НАСТРОЙКА СТРАНИЦЫ
# ============================================
st.set_page_config(
    page_title="MVP: Динамическое ценообразование",
    page_icon="🚀",
    layout="wide"
)

st.title("MVP: Алгоритм динамического ценообразования")
st.markdown("**Спринт 4:** Генератор + Модель + Интерфейс + Симуляция времени")
st.markdown("---")

# ============================================
# 1. ГЕНЕРАТОР ДАННЫХ
# ============================================
@st.cache_data
def generate_data(days=100, products=5, seed=42):
    """
    Генерирует исторические данные
    Формула из задания: Продажи = 100 - 2 * Цена + Случайный_Шум
    """
    np.random.seed(seed)
    
    data = []
    start_date = datetime(2024, 1, 1)
    
    for product_id in range(1, products + 1):
        base_price = 50 + product_id * 10  # 60, 70, 80, 90, 100
        
        for day in range(days):
            current_date = start_date + timedelta(days=day)
            
            # Наша цена
            price = base_price + np.random.normal(0, 5)
            price = max(round(price, 2), 10)
            
            # Цена конкурента (для правил)
            competitor_price = price * np.random.uniform(0.8, 1.2)
            competitor_price = max(round(competitor_price, 2), 10)
            
            # Продажи по формуле из задания
            sales = 100 - 2 * price + np.random.normal(0, 5)
            sales = max(int(round(sales)), 0)
            
            data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'product_id': product_id,
                'product_name': f'Товар {product_id}',
                'price': price,
                'competitor_price': competitor_price,
                'sales': sales
            })
    
    return pd.DataFrame(data)

# ============================================
# 2. МОДЕЛЬ ДЛЯ ОПТИМАЛЬНОЙ ЦЕНЫ
# ============================================
def calculate_optimal_price_model(product_data):
    """
    Расчет оптимальной цены через максимизацию Цена * (A - B * Цена)
    Спрос = A - B * Цена
    Оптимум: A / (2B)
    """
    prices = product_data['price'].values
    sales = product_data['sales'].values
    
    if len(prices) > 1:
        # Регрессия через numpy (метод наименьших квадратов)
        X = np.vstack([prices, np.ones(len(prices))]).T
        try:
            beta = np.linalg.lstsq(X, sales, rcond=None)[0]
            B = -beta[0]  # Коэффициент при цене
            A = beta[1]   # Свободный член
            
            if B > 0:
                optimal = A / (2 * B)
            else:
                optimal = prices.mean()
        except:
            # Если ошибка, используем упрощенный подход
            A = 100
            B = 2
            optimal = A / (2 * B)
    else:
        A = 100
        B = 2
        optimal = A / (2 * B)
    
    return {
        'A': round(A, 2),
        'B': round(B, 2),
        'optimal_price': round(optimal, 2)
    }

# ============================================
# 3. ПРАВИЛА ЦЕНООБРАЗОВАНИЯ (ИЗ ЗАДАНИЯ)
# ============================================
def apply_rules(current_price, competitor_price, previous_sales):
    """
    Применяет два правила из задания:
    1. Если цена конкурента ниже нашей на 10% -> снижаем нашу на 5%
    2. Если продаж вчера не было -> снижаем цену на 1 рубль
    """
    new_price = current_price
    rule = "Без изменений"
    
    # Правило 1
    if competitor_price <= current_price * 0.9:
        new_price = round(current_price * 0.95, 2)
        rule = "Правило 1: конкурент дешевле на 10%"
    
    # Правило 2
    elif previous_sales == 0:
        new_price = round(current_price - 1, 2)
        rule = "Правило 2: не было продаж вчера"
    
    return new_price, rule

# ============================================
# 4. СИМУЛЯЦИЯ ВРЕМЕНИ
# ============================================
def simulate_future(df, product_id, days_ahead=7):
    """
    Симуляция работы алгоритма: цена -> продажи -> новая цена
    """
    # Данные для товара
    product_df = df[df['product_id'] == product_id].copy()
    product_df['date'] = pd.to_datetime(product_df['date'])
    product_df = product_df.sort_values('date')
    
    # Последний день истории
    last_row = product_df.iloc[-1]
    last_date = pd.to_datetime(last_row['date'])
    
    # Текущие значения
    current_price = last_row['price']
    current_sales = last_row['sales']
    
    # Результаты симуляции
    simulation = []
    
    for day in range(1, days_ahead + 1):
        sim_date = last_date + timedelta(days=day)
        
        # Генерируем цену конкурента для этого дня
        competitor_price = round(current_price * np.random.uniform(0.8, 1.2), 2)
        
        # Применяем правила для получения новой цены
        new_price, rule_applied = apply_rules(
            current_price, 
            competitor_price, 
            current_sales
        )
        
        # Прогнозируем продажи при новой цене (по формуле из задания)
        predicted_sales = 100 - 2 * new_price + np.random.normal(0, 5)
        predicted_sales = max(int(round(predicted_sales)), 0)
        
        # Выручка
        current_revenue = current_price * current_sales
        predicted_revenue = new_price * predicted_sales
        
        # Сохраняем шаг
        simulation.append({
            'day': day,
            'date': sim_date.strftime('%Y-%m-%d'),
            'current_price': current_price,
            'competitor_price': competitor_price,
            'current_sales': current_sales,
            'new_price': new_price,
            'predicted_sales': predicted_sales,
            'rule_applied': rule_applied,
            'current_revenue': round(current_revenue, 2),
            'predicted_revenue': round(predicted_revenue, 2),
            'revenue_change': round(predicted_revenue - current_revenue, 2),
            'revenue_change_percent': round((predicted_revenue - current_revenue) / current_revenue * 100, 1) if current_revenue > 0 else 0
        })
        
        # Обновляем для следующего шага
        current_price = new_price
        current_sales = predicted_sales
    
    return pd.DataFrame(simulation)

# ============================================
# 5. ИНИЦИАЛИЗАЦИЯ ДАННЫХ
# ============================================
if 'data_generated' not in st.session_state:
    with st.spinner("Генерация начальных данных..."):
        st.session_state.df = generate_data(days=100, products=5)
        st.session_state.data_generated = True

# ============================================
# 6. БОКОВАЯ ПАНЕЛЬ
# ============================================
with st.sidebar:
    st.header("УПРАВЛЕНИЕ")
    
    st.subheader("1. Параметры данных")
    if st.button("🔄 Сгенерировать новые данные"):
        st.cache_data.clear()
        st.session_state.df = generate_data(
            days=st.session_state.get('days', 100),
            products=st.session_state.get('products', 5)
        )
        st.rerun()
    
    col1, col2 = st.columns(2)
    with col1:
        days = st.number_input("Дней истории:", 50, 200, 100)
    with col2:
        products = st.number_input("Товаров:", 3, 10, 5)
    
    if days != st.session_state.get('days') or products != st.session_state.get('products'):
        st.session_state.days = days
        st.session_state.products = products
        st.session_state.df = generate_data(days=days, products=products)
        st.rerun()
    
    st.markdown("---")
    st.subheader("2. Выбор товара")
    product_id = st.selectbox(
        "Товар для анализа:",
        options=range(1, st.session_state.df['product_id'].nunique() + 1),
        format_func=lambda x: f"Товар {x}"
    )
    
    st.markdown("---")
    st.subheader("3. Симуляция")
    sim_days = st.slider("Дней симуляции:", 3, 30, 7)
    
    if st.button("▶️ Запустить симуляцию", type="primary"):
        st.session_state.run_sim = True
        st.session_state.sim_results = simulate_future(
            st.session_state.df, 
            product_id, 
            sim_days
        )

# ============================================
# 7. ОСНОВНОЙ КОНТЕНТ
# ============================================
df = st.session_state.df

# Статистика
st.header("Общая статистика")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Всего записей", f"{len(df):,}")
with col2:
    st.metric("Товаров", df['product_id'].nunique())
with col3:
    st.metric("Дней в истории", df['date'].nunique())
with col4:
    st.metric("Средняя цена", f"{df['price'].mean():.2f} ₽")

st.markdown("---")

# ============================================
# 8. АНАЛИЗ ВЫБРАННОГО ТОВАРА
# ============================================
st.header(f"Анализ товара {product_id}")

# Фильтруем данные
product_df = df[df['product_id'] == product_id].copy()
product_df['date'] = pd.to_datetime(product_df['date'])
product_df = product_df.sort_values('date')

# Рассчитываем оптимальную цену
optimal = calculate_optimal_price_model(product_df)

# Метрики
col1, col2, col3, col4 = st.columns(4)
with col1:
    current_avg_price = product_df['price'].mean()
    st.metric("Текущая средняя цена", f"{current_avg_price:.2f} ₽")
with col2:
    st.metric(
        "Оптимальная цена",
        f"{optimal['optimal_price']} ₽",
        delta=f"{optimal['optimal_price'] - current_avg_price:.2f} ₽"
    )
with col3:
    st.metric("Коэф. A (спрос)", optimal['A'])
with col4:
    st.metric("Коэф. B (эластичность)", optimal['B'])

# Графики
fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=("Зависимость продаж от цены", "Функция выручки")
)

# График 1: Точки данных
fig.add_trace(
    go.Scatter(
        x=product_df['price'],
        y=product_df['sales'],
        mode='markers',
        name='Исторические данные',
        marker=dict(color='blue', opacity=0.5, size=6)
    ),
    row=1, col=1
)

# Линия регрессии
price_range = np.linspace(
    max(10, product_df['price'].min() * 0.7),
    product_df['price'].max() * 1.3,
    100
)
demand_range = optimal['A'] - optimal['B'] * price_range

fig.add_trace(
    go.Scatter(
        x=price_range,
        y=demand_range,
        mode='lines',
        name=f'Спрос = {optimal["A"]} - {optimal["B"]}×Цена',
        line=dict(color='red', width=2, dash='dash')
    ),
    row=1, col=1
)

# Точка оптимума
fig.add_trace(
    go.Scatter(
        x=[optimal['optimal_price']],
        y=[optimal['A'] - optimal['B'] * optimal['optimal_price']],
        mode='markers',
        name='Оптимальная цена',
        marker=dict(color='green', size=15, symbol='star')
    ),
    row=1, col=1
)

# График 2: Выручка
revenue = price_range * (optimal['A'] - optimal['B'] * price_range)
fig.add_trace(
    go.Scatter(
        x=price_range,
        y=revenue,
        mode='lines',
        name='Выручка',
        line=dict(color='green', width=3)
    ),
    row=1, col=2
)

# Текущая точка
current_revenue = current_avg_price * (optimal['A'] - optimal['B'] * current_avg_price)
fig.add_trace(
    go.Scatter(
        x=[current_avg_price],
        y=[current_revenue],
        mode='markers',
        name='Текущая',
        marker=dict(color='blue', size=12)
    ),
    row=1, col=2
)

# Оптимум на графике выручки
opt_revenue = optimal['optimal_price'] * (optimal['A'] - optimal['B'] * optimal['optimal_price'])
fig.add_trace(
    go.Scatter(
        x=[optimal['optimal_price']],
        y=[opt_revenue],
        mode='markers',
        name='Оптимум',
        marker=dict(color='green', size=12)
    ),
    row=1, col=2
)

fig.update_layout(height=500, showlegend=True)
fig.update_xaxes(title_text="Цена (₽)", row=1, col=1)
fig.update_yaxes(title_text="Продажи", row=1, col=1)
fig.update_xaxes(title_text="Цена (₽)", row=1, col=2)
fig.update_yaxes(title_text="Выручка (₽)", row=1, col=2)

st.plotly_chart(fig, use_container_width=True)

# ============================================
# 9. РЕЗУЛЬТАТЫ СИМУЛЯЦИИ
# ============================================
st.markdown("---")
st.header("⏱️ Симуляция работы алгоритма")

if 'sim_results' in st.session_state and st.session_state.get('run_sim', False):
    sim_df = st.session_state.sim_results
    
    # Таблица с результатами
    st.subheader("Детали по дням")
    
    # Форматируем для отображения
    display_df = sim_df[['day', 'date', 'current_price', 'competitor_price', 
                        'current_sales', 'new_price', 'predicted_sales', 
                        'rule_applied', 'revenue_change_percent']].copy()
    
    display_df.columns = ['День', 'Дата', 'Текущая цена', 'Цена конкурента',
                         'Продажи сегодня', 'Новая цена', 'Прогноз продаж',
                         'Примененное правило', 'Изменение выручки %']
    
    st.dataframe(display_df, use_container_width=True)
    
    # График симуляции
    st.subheader("Динамика цен и продаж")
    
    fig_sim = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Цены
    fig_sim.add_trace(
        go.Scatter(x=sim_df['day'], y=sim_df['current_price'],
                  mode='lines+markers', name='Текущая цена',
                  line=dict(color='blue', width=2)),
        secondary_y=False
    )
    
    fig_sim.add_trace(
        go.Scatter(x=sim_df['day'], y=sim_df['new_price'],
                  mode='lines+markers', name='Новая цена (рекомендация)',
                  line=dict(color='green', width=2, dash='dash')),
        secondary_y=False
    )
    
    fig_sim.add_trace(
        go.Scatter(x=sim_df['day'], y=sim_df['competitor_price'],
                  mode='lines+markers', name='Цена конкурента',
                  line=dict(color='red', width=1, dash='dot')),
        secondary_y=False
    )
    
    # Продажи
    fig_sim.add_trace(
        go.Scatter(x=sim_df['day'], y=sim_df['current_sales'],
                  mode='lines+markers', name='Продажи',
                  line=dict(color='orange', width=2)),
        secondary_y=True
    )
    
    fig_sim.update_xaxes(title_text="День симуляции")
    fig_sim.update_yaxes(title_text="Цена (₽)", secondary_y=False)
    fig_sim.update_yaxes(title_text="Продажи", secondary_y=True)
    
    st.plotly_chart(fig_sim, use_container_width=True)
    
    # Итоговый результат
    st.subheader("Итог симуляции")
    
    total_current = sim_df['current_revenue'].sum()
    total_predicted = sim_df['predicted_revenue'].sum()
    total_change = ((total_predicted - total_current) / total_current * 100) if total_current > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Суммарная выручка (текущая)", f"{total_current:.2f} ₽")
    with col2:
        st.metric("Суммарная выручка (прогноз)", f"{total_predicted:.2f} ₽")
    with col3:
        st.metric("Общее изменение", f"{total_change:+.1f}%")
    
    # Статистика по правилам
    rule_stats = sim_df['rule_applied'].value_counts()
    st.subheader("📊 Статистика применения правил")
    
    for rule, count in rule_stats.items():
        st.write(f"- {rule}: {count} раз ({count/len(sim_df)*100:.0f}%)")

else:
    st.info("Выберите товар и нажмите 'Запустить симуляцию' в боковой панели")

# ============================================
# 10. ИНФОРМАЦИЯ О ПРОЕКТЕ
# ============================================
st.markdown("---")
with st.expander("ℹ️ О проекте"):
    st.markdown("""
    ### Спринт 4: MVP
    
    **Компоненты:**
    - Генератор данных (100 дней, 5 товаров, Продажи = 100 - 2×Цена + Шум)
    - Модель оптимальной цены (максимизация Цена × (A - B×Цена))
    - Правила ценообразования из задания
    - Интерфейс на Streamlit
    - Симуляция времени
    
    **Как работает симуляция:**
    1. Берем последний день истории
    2. Применяем правила → получаем новую цену
    3. Генерируем продажи при новой цене
    4. Повторяем для следующего дня
    """)