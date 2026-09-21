-- SQL Server 2022 — ejemplo institucional (§12). Ajustar tabla/columnas reales.
-- Principio: la app NUNCA concatena SQL; solo ejecuta este SP con parámetro.
CREATE OR ALTER PROCEDURE dbo.sp_consultar_predio_por_nombre
    @NombreCompleto NVARCHAR(100)
AS
BEGIN
    SET NOCOUNT ON;
    -- Búsqueda exacta primero, luego LIKE controlado (sin concatenar en Python)
    SELECT TOP 10
        nombre          AS nombre,
        codigo_predio   AS codigo_predio,
        sector          AS sector,
        area            AS area,
        estado          AS estado
    FROM dbo.Predios
    WHERE nombre = @NombreCompleto
       OR nombre LIKE '%' + @NombreCompleto + '%'
    ORDER BY CASE WHEN nombre = @NombreCompleto THEN 0 ELSE 1 END;
END
GO
-- Permiso mínimo: solo EXEC para el usuario de la app
-- GRANT EXECUTE ON dbo.sp_consultar_predio_por_nombre TO app_juchlm_ro;
