# 🏥 PROYECTO_UNIDAD1IA - Chatbot RAG Inteligente

Repositorio: [https://github.com/pavez845/ChatBot_Ev1](https://github.com/pavez845/ChatBot_Ev1)

## Descripción General

Este proyecto implementa un asistente virtual basado en IA para el Hospital Barros Luco, utilizando técnicas de Retrieval-Augmented Generation (RAG) y modelos de lenguaje (LLMs). El sistema responde consultas médicas y administrativas, recuperando información relevante de documentos internos y generando respuestas precisas y empáticas.

## Características Principales

- Búsqueda híbrida: semántica y por palabras clave.
- Evaluación automática de calidad (fidelidad, relevancia, precisión).
- Interfaz web profesional con Streamlit.
- Panel de métricas y reportes.
- Gestión y edición de documentos hospitalarios.
- Exportación de datos para análisis externo.

## Instalación

### Prerrequisitos
- Python 3.8 o superior
- Acceso a internet
- Token de GitHub Models y/o OpenAI
- Clave de LangSmith (opcional)

### Pasos
1. **Clona el repositorio**
   ```bash
   git clone https://github.com/pavez845/ChatBot_Ev1
   cd proyecto_unidad1IA
   ```
2. **Crea y activa el entorno virtual**
   ```powershell
   python -m venv entorno
   .\entorno\Scripts\Activate.ps1
   ```
3. **Instala las dependencias**
   ```powershell
   pip install -r requirements.txt
   ```
4. **Configura las variables de entorno**
   - Crea un archivo `.env` en la raíz con tus credenciales:
     ```
     GITHUB_TOKEN=tu_token_aqui
     OPENAI_BASE_URL=https://models.inference.ai.azure.com
     OPENAI_EMBEDDINGS_URL=https://models.github.ai/inference
     LANGSMITH_TRACING=true
     LANGSMITH_API_KEY=tu_clave_langsmith
     LANGSMITH_PROJECT=chatbot-universitario
     ```
5. **Ejecuta la aplicación**
   ```powershell
   streamlit run main_rag_evaluation.py
   ```

## Uso
- Accede a la interfaz web para realizar consultas médicas y administrativas.
- Edita y gestiona documentos hospitalarios desde la pestaña correspondiente.
- Visualiza métricas y reportes en tiempo real.
- Exporta datos en formato JSON o CSV para análisis externo.

## Estructura del Proyecto
```
proyecto_unidad1IA/
├── main_rag_evaluation.py   # Código principal del chatbot
├── requirements.txt         # Dependencias
├── .env                     # Variables de entorno
├── README.md                # Este documento
├── chroma_db/               # Base de datos vectorial
├── entorno/                 # Entorno virtual
```

## Documentación Técnica
- El sistema utiliza LangChain y OpenAI para la gestión de agentes y recuperación de información.
- Los documentos hospitalarios se almacenan y procesan en una base de datos vectorial (ChromaDB).
- El pipeline RAG integra búsqueda híbrida y generación de respuestas con métricas automáticas.
- El código está modularizado y documentado para facilitar su mantenimiento.

## Pruebas y Evidencia
- Incluye scripts y funciones para evaluación sistemática de calidad.
- Los resultados pueden exportarse y analizarse en herramientas externas.

## Consideraciones Éticas y Académicas
- El uso de IA se limita a apoyo en redacción y generación de diagramas; las decisiones técnicas y reflexiones son propias del equipo.
- Todo contenido generado con IA fue revisado y validado.
- Las reflexiones individuales se redactan sin apoyo de IA.
- Citación de IA según normativa institucional.

## Referencias
- LangChain Documentation: https://python.langchain.com/
- OpenAI API Documentation: https://platform.openai.com/docs/
- Guía ética IA Duoc UC: https://bibliotecas.duoc.cl/ia