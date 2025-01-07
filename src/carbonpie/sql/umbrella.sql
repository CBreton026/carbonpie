INSERT INTO manual_sums (
    source,
    scenario,
    country,
    category,
    entity,
    unit,
    year,
    value,
    count
)
SELECT source, scenario, 'UMBRELLA', category, entity, unit, year, SUM(value), COUNT(value)
FROM data
WHERE entity = ?
AND country IN (
    SELECT iso_alpha3_code
    FROM countries
    WHERE umbrella=1
) 
GROUP BY scenario, year;