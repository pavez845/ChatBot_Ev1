# 🤖 Chatbot RAG Inteligente

Un chatbot avanzado con Retrieval-Augmented Generation (RAG) que combina búsqueda semántica, evaluación de calidad y un sistema híbrido de memoria.

## ✨ Características Principales

- 🧠 **Búsqueda Híbrida**: Combina búsqueda semántica (70%) y por palabras clave (30%)
- 📊 **Sistema de Evaluación Completo**: Métricas de fidelidad, relevancia y precisión
- 🎨 **Interfaz Profesional**: Dashboard con pestañas organizadas y diseño moderno
- 📈 **Analytics en Tiempo Real**: Visualizaciones interactivas con Plotly
- 🔍 **Filtrado Inteligente**: Detecta automáticamente fuentes relevantes vs conocimiento general
- 🧹 **Gestión de Documentos**: Limpieza automática de placeholders y contenido irrelevante
- 📚 **Base de Vectores**: ChromaDB con embeddings de OpenAI
- 🌡️ **Configuración Optimizada**: Temperatura 0.1 para respuestas consistentes

## 🚀 Características Principales

### 💬 **Conversación Natural**
- Respuestas contextuales basadas en el historial de la conversación
- Detección automática de preguntas simples vs. complejas
- Sistema de memoria híbrida que mantiene contexto relevante

### 📊 **Panel de Métricas Integrado**
- **Métricas en tiempo real**: Tiempos de respuesta, tipos de preguntas, conceptos aprendidos
- **Gráficos interactivos**: Visualización clara del uso y rendimiento
- **Análisis histórico**: Conexión con LangSmith para métricas globales
- **Exportación de datos**: Descarga tus métricas para análisis posterior

### 🎓 **Aprendizaje Automático**
- Extrae automáticamente información valiosa de las conversaciones
- Almacena nuevos conceptos en la base de vectores
- Sistema de transparencia que muestra las fuentes de cada respuesta

### 🔧 **Configuración Flexible**
- Controles granulares de memoria (visual, interna, aprendizaje)
- Posibilidad de agregar información personalizada
- Reset selectivo de diferentes componentes

## 🛠️ Tecnologías Utilizadas

### Inteligencia Artificial
- **LangChain**: Framework para aplicaciones con LLMs
- **GPT-4o-mini**: Modelo de lenguaje de última generación
- **Embeddings**: Vectorización semántica de texto

### Base de Datos
- **ChromaDB**: Base de datos vectorial para búsqueda semántica
- **Almacenamiento persistente**: Mantiene el conocimiento entre sesiones

### Interfaz y Visualización
- **Streamlit**: Framework para aplicaciones web interactivas
- **Plotly**: Gráficos interactivos y dinámicos
- **Pandas**: Análisis y manipulación de datos

### Monitoreo y Métricas
- **LangSmith**: Plataforma de observabilidad para LLMs
- **Métricas locales**: Sistema propio de estadísticas en tiempo real

## 📦 Instalación

### Prerrequisitos
- Python 3.8 o superior
- Token de GitHub Models (gratuito)
- Clave de LangSmith (opcional, para métricas avanzadas)

### Pasos de Instalación

1. **Clona o descarga el proyecto**
   ```bash
   git clone [url-del-proyecto]
   cd chatbot-inteligente
   ```

2. **Crea un entorno virtual**
   ```bash
   python -m venv venv
   
   # En Windows:
   venv\\Scripts\\activate
   
   # En macOS/Linux:
   source venv/bin/activate
   ```

3. **Instala las dependencias**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configura las variables de entorno**
   
   Crea un archivo `.env` en la raíz del proyecto:
   ```env
   # GitHub Models (REQUERIDO)
   GITHUB_TOKEN=tu_token_aqui
   OPENAI_BASE_URL=https://models.inference.ai.azure.com
   OPENAI_EMBEDDINGS_URL=https://models.github.ai/inference
   
   # LangSmith (OPCIONAL - para métricas avanzadas)
   LANGSMITH_TRACING=true
   LANGSMITH_API_KEY=tu_clave_langsmith_aqui
   LANGSMITH_PROJECT=chatbot-inteligente
   ```

5. **Ejecuta el chatbot**
   ```bash
   streamlit run main.py
   ```

## 🔑 Obtener Tokens Gratuitos

### GitHub Models Token
1. Ve a [GitHub Settings > Developer settings > Personal access tokens](https://github.com/settings/tokens)
2. Crea un nuevo token con los permisos necesarios
3. Copia el token y agrégalo al archivo `.env`

### LangSmith (Opcional)
1. Regístrate en [LangSmith](https://smith.langchain.com)
2. Crea un nuevo proyecto
3. Obtén tu API key desde la configuración

## 📊 Usando el Panel de Métricas

El chatbot incluye un panel de métricas completo accesible desde la pestaña "📊 Métricas":

### **Sesión Actual**
- Número de conversaciones realizadas
- Tiempo promedio de respuesta
- Conceptos aprendidos automáticamente
- Distribución de preguntas (simples vs. complejas)

### **Proyecto Global** (requiere LangSmith)
- Historial completo de uso
- Métricas de rendimiento a largo plazo
- Análisis de costos y tokens utilizados

### **Análisis Detallado**
- Configuración actual del sistema
- Enlaces a recursos útiles
- Exportación de datos para análisis externo

## 🎮 Cómo Usar el Chatbot

### Conversación Básica
1. Escribe tu pregunta en el chat
2. El chatbot detectará automáticamente si necesita razonamiento simple o complejo
3. Recibirás una respuesta contextual con indicadores de fuente

### Control de Memoria
- **🗑️ Limpiar Chat Completo**: Elimina todo el historial y memoria
- **🧠 Solo Memoria**: Limpia solo la memoria interna, mantiene historial visual
- **👁️ Solo Visual**: Limpia solo lo que ves, mantiene memoria interna

### Reset Avanzado
- **🧠 Reset Aprendizaje**: Elimina solo el conocimiento aprendido automáticamente
- **💥 Reset TOTAL**: Restaura el chatbot a su estado inicial completo

### Agregar Información
Usa la sección "📝 Agregar Información" para incluir datos específicos que quieres que el chatbot recuerde.

## 🔍 Entendiendo las Fuentes

El chatbot es transparente sobre sus fuentes de información:

- **📚 Base de Datos**: Información que agregaste o aprendió automáticamente
- **🧠 Conocimiento General**: Información del modelo GPT-4o-mini
- **Indicadores visuales**: Cada respuesta muestra claramente su origen

## 🛠️ Personalización

### Modificar la Base de Conocimientos
Edita la función `create_sample_chunks()` en `main.py` para incluir tu información específica.

### Ajustar el Comportamiento
- Cambia la temperatura en la clase `ChatbotInteligente`
- Modifica los prompts en `get_custom_prompt()`
- Personaliza los criterios de preguntas complejas en `is_complex_question()`

## � Estructura del Proyecto

```
chatbot-inteligente/
├── main.py              # Aplicación principal
├── .env                 # Variables de entorno (crear)
├── requirements.txt     # Dependencias del proyecto
├── README.md           # Este archivo
├── chroma_db/          # Base de datos vectorial (se crea automáticamente)
└── venv/               # Entorno virtual (se crea con la instalación)
```

## 🤝 Contribuciones

Este proyecto está diseñado para ser extensible y personalizable. Algunas ideas para mejoras:

- Integración con diferentes modelos de IA
- Soporte para archivos PDF o documentos
- Interfaz de voz
- Integración con APIs externas
- Mejoras en el sistema de métricas

## 📄 Licencia

Este proyecto es de código abierto. Siéntete libre de usarlo, modificarlo y distribuirlo según tus necesidades.

## 🆘 Soporte

Si encuentras algún problema:

1. Verifica que todas las dependencias estén instaladas
2. Asegúrate de que el token de GitHub sea válido
3. Revisa que el archivo `.env` esté configurado correctamente
4. Consulta los logs en la consola para errores específicos

---

**¡Disfruta conversando con tu chatbot inteligente!** 🚀
- 📊 **Sistema de chunks** preparado para expansión
- 🔗 **Integración** con Azure OpenAI y GitHub Models

# 🏥 Chatbot RAG Inteligente - Hospital Barros Luco

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
   git clone <URL-del-repositorio>
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