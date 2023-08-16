/* 

Written by: Nicholas Fasano 
Last Edited: 01/18/2023
Description: Combine movie titles with ratings, cast/crew names, movie year, genre, runtime info
             into a single table and then output the result as a CSV file to be analyzed in Python

*/

/* display information from all tables in the database */
SELECT tconst, STRING_AGG(nconst,','), STRING_AGG(job,',')
FROM titlePrincipals
GROUP BY tconst 


SELECT COUNT(*)
FROM titlePrincipals

SELECT TOP(100) *
FROM nameBasics
ORDER BY nconst DESC

SELECT COUNT(*)
FROM titleBasics
WHERE primaryTitle LIKE 'Titanic'

SELECT COUNT(*)
FROM titleBasics
WHERE tconst LIKE 'tt0120338'

SELECT TOP(10) *
FROM titleCrew
WHERE tconst LIKE 'tt10817940'


SELECT COUNT(*)
FROM titleRatings

-- DROP TABLE #titleCrew_nameBasics_combined
-- DROP TABLE #titleBasics_titleRatings_combined
-- DROP TABLE #titlePrincipals_nameBasics_combined

/* Create temp table #titleCrew_nameBasics_combined combining titleCrew and nameBasic tables */
-- takes about 10 minutes to execute
WITH CTE_TEMP AS
(
	SELECT titleCrew.tconst, cs.Value AS 'directorsAll', css.Value AS 'writersAll' --SplitData
	FROM titleCrew
	CROSS APPLY STRING_SPLIT(titleCrew.directors, ',') AS cs
	CROSS APPLY STRING_SPLIT(titleCrew.writers, ',') AS css
)
SELECT CTE_TEMP.tconst,
direct.primaryName AS directorName, direct.birthYear AS directorBirthYear, direct.deathYear AS directorDeathYear,
writer.primaryName AS writerName, writer.birthYear AS writerBirthYear, writer.deathYear AS writerDeathYear
INTO #titleCrew_nameBasics_combined
FROM CTE_TEMP
LEFT JOIN nameBasics AS direct ON direct.nconst = CTE_TEMP.directorsAll
LEFT JOIN nameBasics AS writer ON writer.nconst = CTE_TEMP.writersAll

-- takes 
DROP TABLE #titleCrew_nameBasics_condense


SELECT  tconst, STRING_AGG(CONVERT(NVARCHAR(max),directorName),',') AS directorName, STRING_AGG(CONVERT(NVARCHAR(max),directorBirthYear),',') AS directorBirthYear, 
		       STRING_AGG(CONVERT(NVARCHAR(max),directorDeathYear),',') AS directorDeathYear, STRING_AGG(CONVERT(NVARCHAR(max),writerName),',') AS writerName, 
			   STRING_AGG(CONVERT(NVARCHAR(max),writerBirthYear),',') AS writerBirthYear, STRING_AGG(CONVERT(NVARCHAR(max),writerDeathYear),',') AS writerDeathYear
INTO #titleCrew_nameBasics_condense
FROM #titleCrew_nameBasics_combined
GROUP BY tconst


SELECT TOP(1000) *
FROM #titleCrew_nameBasics_condense

SELECT TOP(100) *
FROM #titleCrew_nameBasics_combined

/* Create temp table #titleBasics_titleRatings_combined combining titleBasics and titleRatings */
-- takes ~30seconds to execute
SELECT titleBasics.tconst, titleBasics.titleType, titleBasics.primaryTitle, titleBasics.startYear,
titleBasics.endYear, titleBasics.runtimeMinutes, titleBasics.genres, 
titleRatings.averageRating, titleRatings.numVotes
INTO #titleBasics_titleRatings_combined
FROM titleBasics
LEFT JOIN titleRatings ON titleBasics.tconst=titleRatings.tconst

/* Create temp table #titlePrincipals_nameBasics_combined combining titlePrincipals and nameBasics */
-- takes ~4min30sec to execute 
SELECT tp.tconst, nb.primaryName, nb.birthYear, nb.deathYear, tp.category
INTO #titlePrincipals_nameBasics_combined
FROM titlePrincipals AS tp
LEFT JOIN nameBasics AS nb ON nb.nconst = tp.nconst

/* finally join all three temp tables on tconst and create table titleBasics_titleRatings_titleCrew_nameBasics_combined
   export table as CSV file using SQL server export wizard */
SELECT T1.tconst, T1.primaryTitle, T1.titleType, T1.startYear, T1.endYear, T1.runtimeMinutes,
T1.genres, T1.averageRating, T1.numVotes, T2.directorName, T2.directorBirthYear, 
T2.directorDeathYear, T2.writerName, T2.writerBirthYear, T2.writerDeathYear,
T3.primaryName, T3.birthYear AS primaryNameBirthYear, T3.deathYear AS primaryNameDeathYear, T3.category AS primaryNameCategory
INTO titleBasics_titleRatings_titleCrew_nameBasics_titlePrincipals_combined
FROM #titleBasics_titleRatings_combined AS T1
LEFT JOIN #titleCrew_nameBasics_combined AS T2 ON T1.tconst = T2.tconst
LEFT JOIN #titlePrincipals_nameBasics_combined AS T3 ON T1.tconst = T3.tconst

SELECT TOP(100) T1.tconst, T1.primaryTitle, T1.titleType, T1.startYear, T1.endYear, T1.runtimeMinutes,
T1.genres, T1.averageRating, T1.numVotes, T2.directorName, T2.directorBirthYear, 
T2.directorDeathYear, T2.writerName, T2.writerBirthYear, T2.writerDeathYear,
T3.primaryName, T3.birthYear AS primaryNameBirthYear, T3.deathYear AS primaryNameDeathYear, T3.category AS primaryNameCategory
FROM #titleBasics_titleRatings_combined AS T1
FULL OUTER JOIN #titleCrew_nameBasics_condense AS T2 ON T1.tconst = T2.tconst
FULL OUTER JOIN #titlePrincipals_nameBasics_combined AS T3 ON T1.tconst = T3.tconst

SELECT TOP(10) *
FROM titleBasics_titleRatings_titleCrew_nameBasics_titlePrincipals_combined

DROP TABLE titleBasics_titleRatings_titleCrew_nameBasics_titlePrincipals_combined


SELECT TOP(10) *
FROM titleBasics_titleRatings_titleCrew_nameBasics_combined
WHERE averageRating IS NOT NULL 
ORDER BY startYear DESC



SELECT TOP(10) *
FROM (
	
	SELECT tconst, primaryTitle, titleType, startYear, endYear, runtimeMinutes, genres, averageRating, numVotes,
	       directorName, directorBirthYear, directorDeathYear, writerName, writerBirthYear, writerDeathYear,
		   primaryName, primaryNameBirthYear, primaryNameDeathYear, primaryNameCategory,
	       ROW_NUMBER() OVER(PARTITION BY tconst ORDER BY averageRating DESC) AS rn
	FROM titleBasics_titleRatings_titleCrew_nameBasics_combined
	WHERE directorName IS NOT NULL 
) AS a
WHERE rn = 1 AND titleType LIKE 'Movie' AND averageRating IS NOT NULL AND numVotes > 100 AND averageRating > 6
ORDER BY startYear DESC











