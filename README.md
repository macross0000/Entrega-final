# Entrega-final
BSG Institute proyecto final Certified AI/LLM Solution Architect


Documentación Técnica
RAG Assistant para Recursos Humanos

README técnico y guía de arquitectura para GitHub


Versión: 1.0
Fecha: Abril 2026
Clasificación: Documento técnico / académico


Proyecto: Prueba de concepto RAG con ChromaDB, Sentence Transformers y Gemini/Gemma 
1. Control documental
Nombre del documento	Documentación técnica para GitHub - RAG Assistant RH
Versión	1.0
Tipo de documento	README técnico / documentación de arquitectura
Alcance	Prueba de concepto RAG para consultas de Recursos Humanos
Estado	Borrador ejecutivo listo para integración en GitHub
2. Descripción general
Este proyecto implementa una prueba de concepto de un asistente de Recursos Humanos basado en arquitectura RAG, Retrieval Augmented Generation. El objetivo es permitir que un usuario realice preguntas en lenguaje natural sobre documentos internos y que el sistema responda utilizando únicamente la información contenida en dichos documentos.
La solución combina búsqueda semántica, embeddings multilingües, almacenamiento vectorial persistente y generación de respuesta mediante un modelo generativo configurado para entregar respuestas breves, controladas y basadas en contexto.
Objetivo del sistema
Responder preguntas de Recursos Humanos usando documentos internos como fuente de verdad, minimizando respuestas inventadas y estableciendo controles básicos contra instrucciones maliciosas o prompt injection.

3. Caso de uso
El caso de uso principal es un asistente de consulta para Recursos Humanos capaz de responder preguntas sobre información interna documentada, tales como días festivos, políticas internas, preguntas frecuentes, capacitación, información operativa y lineamientos incluidos en documentos Markdown.
El sistema está diseñado para evitar respuestas no fundamentadas. Si la respuesta no se encuentra en el contexto recuperado, el modelo debe responder:
No tengo suficiente información
4. Tecnologías utilizadas
Componente	Tecnología
Entorno de desarrollo	Google Colab
Lenguaje	Python
Interfaz opcional	Streamlit
Base vectorial	ChromaDB
Modelo de embeddings	intfloat/multilingual-e5-large
Modelo generativo	models/gemma-3n-e2b-it
Framework de splitting	LangChain Text Splitters
Formato documental	Markdown
Persistencia	Google Drive / ChromaDB persistente
Seguridad básica	Filtro de prompt injection mediante lista de términos bloqueados
5. Arquitectura funcional
El flujo funcional del sistema puede representarse de la siguiente forma:
Documentos RH
     ↓
Carga de documentos
     ↓
Chunking / división por encabezados Markdown
     ↓
Generación de embeddings
     ↓
Almacenamiento en ChromaDB
     ↓
Pregunta del usuario
     ↓
Limpieza y validación de pregunta
     ↓
Búsqueda semántica en ChromaDB
     ↓
Recuperación de contexto relevante
     ↓
Generación de respuesta con Gemma/Gemini
     ↓
Evaluación básica de confiabilidad
     ↓
Respuesta final al usuario
6. Estructura lógica del código
6.1 Instalación de dependencias
El notebook instala las librerías necesarias para ejecutar el flujo RAG. Las dependencias principales son:
streamlit
chromadb
pathlib
langchain-text-splitters
sentence-transformers
pandas
torch
transformers
google-generativeai
Durante la instalación se observan advertencias de compatibilidad relacionadas con paquetes de opentelemetry. Estas advertencias no necesariamente bloquean la ejecución de la prueba de concepto, pero deben atenderse antes de llevar la solución a un ambiente productivo.
6.2 Configuración de variables principales
El código define rutas, modelo de embeddings, colección de ChromaDB y modelo generativo:
files_path = "./drive/MyDrive/Proyecto_final"
Chroma_path = "/content/drive/MyDrive/Proyecto_final/chroma_db"
Modelo_Emb = "intfloat/multilingual-e5-large"
Nombre_Coleccion = "proyecto final"

modelo_embeddings = SentenceTransformer(Modelo_Emb)
chroma_client = chromadb.PersistentClient(path=Chroma_path)
modelo_llm = genai.GenerativeModel("models/gemma-3n-e2b-it")
7. Componentes principales
7.1 Carga de documentos
El sistema utiliza documentos almacenados en Google Drive. Estos documentos se procesan para extraer contenido textual y metadatos, principalmente el nombre del archivo, que permite mantener trazabilidad básica sobre la fuente documental.
7.2 Chunking de documentos
La función chunkenizar_documentos() divide documentos Markdown utilizando encabezados de nivel 1, 2 y 3. Esto permite conservar una estructura lógica del documento y generar fragmentos útiles para búsqueda semántica.
7.3 Generación de embeddings
El proyecto utiliza el modelo intfloat/multilingual-e5-large para convertir textos y preguntas en vectores semánticos. El código aplica el formato recomendado por modelos E5 usando prefijos query: para preguntas y passage: para textos documentales.
7.4 Almacenamiento en ChromaDB
Los chunks procesados se almacenan en una colección persistente de ChromaDB. Esto permite realizar búsquedas semánticas sobre los documentos indexados y mantener la base vectorial entre ejecuciones.
7.5 Recuperación de documentos relevantes
La función recuperar_documentos_relevantes() recibe una pregunta, genera su embedding y consulta ChromaDB para recuperar los documentos más cercanos semánticamente. Por defecto, recupera los tres documentos más relevantes.
7.6 Protección básica contra prompt injection
La función limpiar_pregunta() detecta frases potencialmente maliciosas o instrucciones orientadas a manipular el comportamiento del modelo, tales como ignora instrucciones, actúa como, system prompt, modo desarrollador, override o jailbreak.
7.7 Generación de respuesta
La función generar_respuesta() construye un prompt controlado. El prompt instruye al modelo para usar únicamente el contexto recuperado, ignorar instrucciones contradictorias, no revelar prompts internos y no inventar información.
7.8 Evaluación de respuesta
La función evaluar_respuesta() solicita al modelo evaluar si la respuesta está sustentada en el contexto. Devuelve 1 cuando la respuesta es correcta y 0 cuando se considera incorrecta o potencialmente alucinada.
7.9 Pipeline RAG
La función rag_pipeline() integra validación de pregunta, recuperación semántica, construcción de contexto, generación de respuesta, evaluación básica y devolución de la respuesta final al usuario.
8. Fragmentos clave del código
8.1 Recuperación semántica
def recuperar_documentos_relevantes(pregunta: str, coleccion, modelo_embeddings, top_n: int = 3):
    query_embedding = modelo_embeddings.encode(["query: " + pregunta])[0]

    resultados = coleccion.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_n
    )

    return resultados["documents"][0]
8.2 Hiperparámetros del modelo generativo
generation_config = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_output_tokens": 150
}
Parámetro	Valor	Propósito
temperature	0.0	Reduce variabilidad y hace las respuestas más determinísticas
top_p	1.0	Permite considerar todo el espacio probable de tokens
max_output_tokens	150	Limita la longitud de la respuesta
9. Instalación y uso
9.1 Clonar repositorio
git clone <URL_DEL_REPOSITORIO>
cd <NOMBRE_DEL_REPOSITORIO>
9.2 Crear entorno virtual
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
9.3 Instalar dependencias
pip install -r requirements.txt
9.4 Configurar API Key
En Google Colab se recomienda almacenar la clave como secreto con el nombre GEMINI_API_KEY. Para ejecución local, se recomienda usar un archivo .env.
GEMINI_API_KEY=your_api_key_here
10. Estructura recomendada del repositorio
rag-rh-assistant/
│
├── app.py
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── data/
│   ├── Capacitacion_RH.md
│   └── Preguntas_Frecuentes.md
│
├── chroma_db/
│   └── .gitkeep
│
├── notebooks/
│   └── hiperparametros_prueba_de_concepto_final.ipynb
│
└── src/
    ├── document_loader.py
    ├── chunking.py
    ├── embeddings.py
    ├── vector_store.py
    ├── rag_pipeline.py
    └── security.py
11. Archivos recomendados para GitHub
11.1 .gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.pyd

# Entornos virtuales
venv/
.env

# Jupyter
.ipynb_checkpoints/

# ChromaDB local
chroma_db/

# Sistema operativo
.DS_Store
Thumbs.db

# Logs
*.log
11.2 requirements.txt
chromadb
sentence-transformers
langchain-text-splitters
google-generativeai
streamlit
python-dotenv
pandas
torch
transformers
12. Limitaciones actuales
Área	Observación
Seguridad	El filtro contra prompt injection es básico y depende de una lista negra.
Gestión de secretos	La API Key debe mantenerse fuera del código fuente.
Evaluación	La validación de respuestas depende del mismo modelo generativo.
Observabilidad	No se observan logs estructurados ni trazabilidad completa.
Manejo de errores	El manejo de excepciones es limitado.
Dependencias	Existen advertencias de compatibilidad en algunos paquetes instalados.
Escalabilidad	El flujo está orientado a Google Colab y no a despliegue productivo.
Gobierno de datos	No se observa clasificación documental ni control de acceso por rol.
13. Recomendaciones de mejora
•	Modularización: Separar el notebook en módulos Python independientes: loader, chunker, retriever, generator, security y app.
•	Seguridad: Agregar validación de entrada, límite de caracteres, filtros más robustos contra prompt injection y logs de consultas bloqueadas.
•	Gobierno de datos: Incorporar clasificación documental, control de acceso por rol, exclusión de datos personales sensibles y política de retención.
•	Observabilidad: Implementar logging estructurado para preguntas, documentos recuperados, confianza de respuesta y errores.
•	Calidad: Agregar pruebas unitarias para las funciones principales del pipeline RAG.
•	Despliegue: Empaquetar la solución en Docker para facilitar ejecución reproducible en plataformas académicas o empresariales.
14. Roadmap sugerido
Fase	Objetivo	Resultado esperado
Fase 1	Limpieza del notebook	Código ordenado y reproducible
Fase 2	Modularización	Separación en archivos .py
Fase 3	Interfaz Streamlit	Aplicación usable por usuarios finales
Fase 4	Seguridad	Validación de entradas, logs y control de secretos
Fase 5	Evaluación	Métricas de precisión, cobertura y alucinación
Fase 6	Despliegue	Docker / plataforma institucional
Fase 7	Gobierno	Control de acceso, auditoría y documentación formal
15. Conclusión ejecutiva
El proyecto demuestra la viabilidad de construir un asistente de Recursos Humanos basado en RAG, capaz de responder preguntas usando documentos internos como fuente de verdad. La solución combina búsqueda semántica con una base vectorial persistente en ChromaDB y generación de respuestas mediante un modelo LLM configurado con baja temperatura para reducir variabilidad.
La prueba de concepto es funcional para validación académica o demostración técnica, pero requiere ajustes de seguridad, modularización, control de dependencias, manejo de secretos y trazabilidad antes de considerarse apta para un entorno empresarial o productivo.
16. Anexo A - README.md listo para GitHub
# RAG Assistant para Recursos Humanos

## Descripción

Este proyecto implementa una prueba de concepto de un asistente de Recursos Humanos basado en arquitectura RAG — Retrieval Augmented Generation.

El sistema permite realizar preguntas en lenguaje natural sobre documentos internos y genera respuestas usando únicamente el contexto recuperado desde una base vectorial.

## Tecnologías

- Python
- Google Colab
- ChromaDB
- Sentence Transformers
- LangChain Text Splitters
- Google Generative AI
- Gemma / Gemini
- Streamlit

## Modelo de embeddings

intfloat/multilingual-e5-large

## Modelo generativo

models/gemma-3n-e2b-it

## Flujo general

1. Carga de documentos.
2. División en chunks.
3. Generación de embeddings.
4. Almacenamiento en ChromaDB.
5. Consulta del usuario.
6. Recuperación semántica.
7. Generación de respuesta.
8. Evaluación básica de confiabilidad.

## Instalación

```bash
git clone <URL_DEL_REPOSITORIO>
cd <NOMBRE_DEL_REPOSITORIO>
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Variables de entorno

Crear archivo `.env`:

```env
GEMINI_API_KEY=your_api_key_here
```

## Ejecución

En Google Colab, ejecutar las celdas en orden.

Para Streamlit:

```bash
streamlit run app.py
```

## Seguridad

El proyecto incluye una función básica para detectar intentos de prompt injection mediante una lista de palabras bloqueadas.

## Limitaciones

- Filtro de seguridad básico.
- Evaluación dependiente del mismo modelo generativo.
- Sin autenticación por usuario.
- Sin control de acceso por rol.
- Sin logs estructurados.
- Pensado como prueba de concepto, no como solución productiva.

## Roadmap

- Modularizar el código.
- Crear aplicación Streamlit.
- Agregar Docker.
- Implementar logging.
- Mejorar controles de seguridad.
- Agregar pruebas unitarias.
- Implementar gobierno de datos.
```
17. Fuente de análisis
Documento elaborado con base en el notebook/código compartido por el usuario en el archivo “Pasted text.txt”, correspondiente a la prueba de concepto hiperparametros_prueba_de_concepto__final.ipynb.
