BUTTONS = {
    '🎬 Кружки': '/video',
    '🖼 Фото': '/photo',
    '✨ Нейроулучшение': '/mode ai',
    '🌿 Бережное улучшение': '/mode safe',
    '🔍 Увеличить ×2': '/scale 2',
    '🔎 Увеличить ×4': '/scale 4',
    '🎯 Центр кадра': '/fit crop',
    '🖼 Весь кадр': '/fit contain',
    '📊 Статус': '/status',
    '⛔ Отмена': '/cancel',
    '❓ Помощь': '/help',
}
KEYBOARD = {
    'keyboard': [list(BUTTONS)[i:i+2] for i in range(0, len(BUTTONS), 2)],
    'resize_keyboard': True,
    'is_persistent': True,
    'input_field_placeholder': 'Видео, фото или кнопка меню',
}
