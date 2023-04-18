[![Raspli Logo](https://user-images.githubusercontent.com/70770455/232812807-a9ad7134-5107-44e5-bf74-e5826e458669.png)](https://raspli.ru)

# Распли

## Идея
Система распознавания лиц может помочь учебным заведениям стать безопаснее и удобнее. Потенциально такая система способна идентифицировать каждого человека в учебном заведении, считывать его внешние параметры (направление внимания, настроение и другие), а затем собирать и обрабатывать данные в реальном времени.

## Детектор

Для детектирования лиц мы используем модель нейросети YOLOv8, которая на данный момент является State-Of-The-Art моделью для детектирования объектов.

### Веса

Обучать большие модели нейросетей достаточно трудоемко, поэтому на данный момент доступна только легкая *n*-версия модели.

- [x] YOLOv8n-face
- [ ] YOLOv8s-face
- [ ] YOLOv8m-face
- [ ] YOLOv8l-face
- [ ] YOLOv8x-face

### Модуль

Для использования детектора достаточно запустить `detector.py` и указать источник детектирования (изображение, видео, видеострим), например:

```bash
python3 detector.py --source /home/fruitourist/Desktop/futurists.jpg
```

Также вы можете указать путь до *весов модели* и *необходимость сохранения результата*.

### Пример результата

![Raspli Detector Example](https://user-images.githubusercontent.com/70770455/232813793-ddff9044-c663-441f-b6ee-c1d7ba26806b.png)