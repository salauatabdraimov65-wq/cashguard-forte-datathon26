# CashGuard — выявление нарушений кассовой дисциплины по видео (ForteBank Datathon 26)

Сервис: https://5gofnn1dvheh3y-8000.proxy.runpod.net (сайт и API: `POST /api/v1/analyze`, `GET /api/v1/jobs/{job_id}`)

Весь код и подробное описание (архитектура, запуск, результаты, нормативы, ограничения) —
в папке [cashguard-github](cashguard-github/README.md).

- `cashguard-github/service/` — API и надстройка модели
- `cashguard-github/src/` — основной пайплайн (детекторы + Qwen3.5-9B, правила)
- `cashguard-github/site/` — веб-интерфейс
- `cashguard-github/notebooks/`, `cashguard-github/tools/` — обучение детектора и проверка устойчивости

Результат на открытой выборке: 85% верных вердиктов (34/40), на зеркальных копиях 79%.
Работает локально, без сторонних AI API.
