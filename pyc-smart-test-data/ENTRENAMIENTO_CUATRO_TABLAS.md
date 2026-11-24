# PyC Smart Test Data Lab: entrenamiento en cuatro tablas (experimental)

## Qué hace
- **Solo Clientes** conserva el modo original: TVAE aprende segmento, región, nivel_riesgo y edad; los descendientes se derivan de patrones empíricos y reglas.
- **Todas las tablas (4 modelos TVAE)** entrena cuatro sintetizadores independientes, en CPU, con las columnas / representaciones:
  - Clientes: segmento, region, nivel_riesgo, edad.
  - Ventas: canal, monto_venta.
  - Facturas: factor_facturacion (monto_total/monto_venta), estado.
  - Pagos: fraccion_pago (monto_pago/monto_total) y monto_pago.
- La cantidad de ventas por cliente, facturas por venta y pagos por factura sigue procediendo de frecuencias observadas. Las claves y montos coherentes se reconstruyen con reglas explícitas. **No aprende automáticamente distribuciones condicionales entre entidades** ni constituye todavía un sintetizador multitabla generativo.
- Se validan claves primarias y foráneas, coincidencia de clientes, estados y conciliación de importes.
- Se preserva el modo original y la descarga de modelos. En el modo completo, el respaldo incluye los cuatro modelos entrenados; puede ser más grande. Los respaldos requieren validación y versiones compatibles, como antes.

## Identidades ficticias
- Faker es_ES genera nombres y nombres de empresa plausibles; direcciones incluyen Zona Demo.
- El correo utiliza `example.invalid`, no un dominio real. Los teléfonos llevan caracteres XXX y no son números marcables.
- Estos datos **no son clientes reales** ni una solución de anonimización. No cargar datos personales sin controles.

## Prueba de aceptación recomendada (antes de entrenar largo)
1. Regenerar datos de demostración y aprobar diccionarios de las cuatro entidades.
2. Entrenar modo **Todas las tablas** con **10 épocas**, esperar a ver el modelo terminado.
3. Generar 50 clientes y comprobar que salen Clientes, Ventas, Facturas y Pagos, sin errores de integridad.
4. Descargar experimento para inspeccionar sus datos; guardar respaldo `.pycmodel`.
5. Cerrar la sesión, importar el respaldo y comprobar generación sin reentrenar.
6. Comparar dos generaciones con el comparador A/B. Las 7 métricas existentes son orientativas y no acreditan privacidad ni consistencia funcional completa.

## Estado de validación
Cambio de código publicado en GitHub; **prueba integral en Streamlit Cloud todavía pendiente**. La integración requiere verificar comportamiento real con SDV, Streamlit y las dependencias fijadas. No iniciar entrenamientos largos hasta completar el ciclo de aceptación.
