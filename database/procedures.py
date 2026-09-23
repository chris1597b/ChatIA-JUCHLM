"""Stored Procedures institucionales (§12). Cero SQL inline en app.py/services."""

SP_CONSULTAR_PREDIO_POR_NOMBRE = "EXEC dbo.sp_consultar_predio_por_nombre @NombreCompleto = ?"

# Padrón real: un DNI puede tener N predios (una fila por predio).
SP_OBTENER_PREDIOS_POR_DNI = "EXEC dbo.sp_ObtenerPrediosPorDNI @Dni = ?"
