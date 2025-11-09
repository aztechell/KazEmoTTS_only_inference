# KazEmoTTS

Эксперемент с [KazEmoTTS](https://github.com/IS2AI/KazEmoTTS) 

## Установка и использование готовых моделей

1. Импортировать проект (Project from version control)   
2. Создать виртуальную среду (желательно с 3.9.13)  
3. Установить pytorch ``` pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128 ```  
4. установить модули
5. скачать тренированную модель (чекпоинт) [https://issai.nu.edu.kz/wp-content/uploads/2024/03/pt_10000.zip](https://issai.nu.edu.kz/wp-content/uploads/2024/03/pt_10000.zip) и (https://issai.nu.edu.kz/wp-content/uploads/2024/03/pre_trained_hf.zip)  
6. Загрузить скачанные файлы EMA_grad_10000.pt и g_01720000 в папку pre_trained  
7. Построить monotonic_align ``` cd model/monotonic_align; python setup.py build_ext --inplace; cd ../.. ```  
8. Запустить через ```.\start.bat```  

Если в папке generated появились аудиофайлы, то значит всё работает и можно экспериментировать дальше.   
Текст задается в файле ```.\filelists\inference_generated.txt```   
Голоса: 0 - M1, 1 - F1, 2 - M2.   
Эмоции: 0 - "angry", 1 - "surprise", 2 - "fear",3 - "happy",4 - "neutral", 5 - "sad".   

