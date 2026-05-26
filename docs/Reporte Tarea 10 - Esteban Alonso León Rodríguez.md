# Reporte Tarea 10 - Esteban Alonso León Rodríguez

## A. Definición de condiciones

#### (A) Condición Experimental
La condición experimental consistirá en un agente virtual multimodal diseñado para atención al cliente. El sistema integrará un modelo lenguaje (Gemini) combinado con un mecanismo de Retrieval-Augmented Generation (RAG) alimentado por tickets históricos y documentación relevante sobre el uso de Mediación Virtual. El agente permitirá interacción tanto por texto como por voz mediante tecnologías de Speech-to-Text (STT) y Text-to-Speech (TTS). Adicionalmente, contará con una representación visual mediante un avatar animado capaz de mostrar estados básicos de interacción, como escucha y respuesta con lip-sync, con el objetivo de incrementar la naturalidad de la interacción. 

#### (B) Condición de Control o Baseline
La condición de control consistirá en una interfaz conversacional basada únicamente en texto que ya existe en la plataforma de Mediación Virtual. Este sistema utiliza un modelo de lenguaje y tiene acceso a una base de conocimiento RAG, pero no cuenta con componentes multimodales como interacción por voz, avatar visual animado y elementos de embodiment. La interacción se realizará exclusivamente mediante mensajes de texto en una interfaz de chat, lo que permite establecer una comparación entre una experiencia tradicional y una experiencia multimodal.

#### Justificación
La comparación entre ambas condiciones permite aislar el impacto de características multimodales y de embodiment del agente virtual sobre la experiencia del usuario. Ambos agentes harían uso de RAGs, por lo que las diferencias observadas en métricas como satisfacción, usabilidad, percepción de naturalidad o eficiencia podrán atribuirse principalmente a la incorporación de voz, representación visual e interacción más "humana". Este diseño experimental permite evaluar si los elementos característicos de un agente virtual multimodal genera un impacto significativo respecto a un chatbot textual tradicional en las métricas.

## B. Matriz de Consistencia Metodológica

| Pregunta de Investigación (RQ) | Variable / Constructo | Instrumento Validado | Tarea Asociada |
|---|---|---|---|
| **RQ1. ¿Cómo impacta el uso de un agente virtual multimodal con voz y embodiment en la satisfacción y usabilidad percibida por los usuarios en comparación con un chatbot tradicional basado en texto?** | Usabilidad percibida | SUS (System Usability Scale) | Resolver una consulta de soporte utilizando el sistema asignado (texto o multimodal) |
|  | Satisfacción del usuario | meCUE / CSAT | Completar interacción completa de atención al cliente |
|  | Presencia social y naturalidad | Godspeed Questionnaire Series | Interactuar con el agente mediante voz y avatar |
| **RQ2. ¿Cómo influye la incorporación de memoria conversacional y capacidades multimodales en la eficiencia de resolución de tareas de atención al cliente?** | Tiempo de resolución | Medición objetiva de tiempo | Resolver una tarea específica de soporte técnico |
|  | Tasa de éxito | Registro de completitud de tarea | Completar exitosamente el escenario planteado |
|  | Carga cognitiva | NASA-TLX | Resolver una tarea con múltiples pasos e instrucciones |

Es importante aclarar qué tarea estarán realizando los participantes, pues esta debe ser la misma en ambas condiciones. Para esto se le estará consultando a expertos por un escenario común de atención al cliente al que ellos se enfrentan. Un caso de ejemplo simple que se les podría poner a los participantes es: "Asuma el rol de un estudiante universitario matriculado a un curso X, que al entrar a Mediación Virtual no le aparece el curso matriculado. Utilice el sistema asignado para resolver su situación."

## C. Justificación

La justificación del diseño del agente con características de embodiment pretende, con esperanza de obtener resultados positivos, que la interacción tenga una mejoría con respecto a un diseño tradicional sólo basado en texto. Justine Cassell plantea que los agentes virtuales no solo se comunican con información mediante lenguaje verbal, sino también mediante señales no verbales como expresiones faciales, mirada, movimientos, etcétera. El avatar no pretende sólo representar visualmente al sistema, sino también actuar como un complemento de la comunicación para apoyar la percepción de presencia social y la naturalidad durante la conversación. Esto resulta especialmente relevante en tareas de atención al cliente, donde la interacción suele percibirse como impersonal o mecánica en sistemas basados únicamente en texto.

Por otro lado, la decisión de diseño de no utilizar un avatar tan realista e inclinarse por uno más caricaturesco y antropomórfico viene con el objetivo de evitar un efecto de uncanny-valley. Considero que para que un avatar hiperrealista sea efectivo, tiene que ser uno muy detallado y bien hecho, lo cual es difícil de encontrar de forma gratuita. Esta decisión se fundamenta en investigaciones relacionadas con el fenómeno de uncanny-valley, las cuales sugieren que avatares excesivamente realistas pueden generar sensaciones de incomodidad o rechazo cuando existen inconsistencias en apariencia o comportamiento. Shin, Kim y Biocca (2019) encontraron que avatares hiperrealistas producen mayores niveles de eeriness en comparación con representaciones caricaturescas o estilizadas. Asimismo, Chattopadhyay y MacDorman (2016) señalan que pequeñas inconsistencias visuales o conductuales en personajes casi humanos pueden intensificar este efecto negativo.

## Referencias

Cassell, J., Bickmore, T., Campbell, L., Vilhjalmsson, H., & Yan, H. (2000). Designing embodied conversational agents. Embodied conversational agents, 29-63.

Chattopadhyay, D., & MacDorman, K. F. (2016). Familiar faces rendered strange: Why inconsistent realism drives characters into the uncanny valley. Journal of Vision, 16(11), 7. https://doi.org/10.1167/16.11.7

Shin, M., Kim, S. J., & Biocca, F. (2019). The uncanny valley: No need for any further judgments when an avatar looks eerie. Computers in Human Behavior, 94, 100–109. https://doi.org/10.1016/j.chb.2019.01.016
