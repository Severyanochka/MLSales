# Прогнозирование заказов с использованием машинного обучения

## Описание

Этот проект нацелен на прогнозирование объема заказов в интернет-магазине одежды на основе исторических данных. Мы используем методы линейной регрессии для создания модели, которая предсказывает объем заказов, основываясь на различных признаках, таких как дата, страна, категория товара и цена.
Данные относятся к интернет-магазину одежды за 2008 год и содержат следующие переменные:

1. **ГОД ВЫПУСКА**: 2008
2. **МЕСЯЦ**: от апреля (4) до августа (8)
3. **ДЕНЬ**: номер дня месяца
4. **ПОРЯДОК**: последовательность кликов в течение одного сеанса
5. **СТРАНА**: страна происхождения IP-адреса (см. список стран выше)
6. **ИДЕНТИФИКАТОР СЕАНСА**: идентификатор сеанса (короткая запись)
7. **СТРАНИЦА 1 (ОСНОВНАЯ КАТЕГОРИЯ)**: основная категория товаров (брюки, юбки, блузки, распродажа)
8. **СТРАНИЦА 2 (МОДЕЛЬ ОДЕЖДЫ)**: код товара (217 товаров)
9. **ЦВЕТ**: цвет изделия (см. список цветов выше)
10. **РАСПОЛОЖЕНИЕ**: расположение фотографии на странице (6 частей)
11. **ФОТОГРАФИЯ МОДЕЛИ**: анфас или профиль
12. **ЦЕНА**: цена в долларах США
13. **ЦЕНА 2**: превышает ли цена товара среднюю цену по категории (да/нет)
14. **СТРАНИЦА**: номер страницы на веб-сайте (от 1 до 5)

## Ошибки и улучшения

После обучения модели была обнаружена ошибка, связанная с нелинейностью данных. Это указывает на необходимость подбора более подходящей модели или дальнейшей настройки гиперпараметров.
