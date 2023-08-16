/* Nicholas Fasano 
06/28/2022
Query IMDB database mainly to practice/learn SQL commands

*/

/* --------------------------------------------- */
/* --------------------------------------------- */
/* --------------------------------------------- */
/* ------------- Stored Procedures ------------- */
/* --------------------------------------------- */
/* --------------------------------------------- */
/* --------------------------------------------- */
-- DROP PROCEDURE [dbo].[movieName]
-- DROP PROCEDURE [dbo].[personName] 
/*
CREATE PROCEDURE movieName
AS
SELECT TOP(5) originalTitle
FROM titleBasics



CREATE PROCEDURE personName
@actorID nvarchar(100)
AS
SELECT TOP(5) *
FROM nameBasics
WHERE nconst = @actorID
*/
/* Explore each table */

/*nameBasics --- name of person in database , their ID (nconst), 
birth year, death year, primary profession, known for title by ID 
Most entries in birth/death year are \N == NULL */
SELECT TOP(10) *
FROM nameBasics 
WHERE primaryName LIKE 'Kate Winslet'

SELECT TOP(10) *
FROM nameBasics 
WHERE primaryName LIKE 'Leonardo Dicaprio'

/* title basics --- movieID (tconst), titleType (tvEpisode, ...) Original title,
primary title, isAdult (Boolean), start year, endyear, runtime [min], genres (>1000 distinct genres) */
SELECT TOP(3) *
FROM titleBasics
--WHERE tconst LIKE 'tt0027511'

/* title crew --- movieID (tconst), directors, and writers */
SELECT TOP(3) *
FROM titleCrew
--WHERE tconst LIKE 'tt0120338'
/* tt0120338 -- titanic movieID */

/* title Episodes -- episodeID (tconst), showID (parentTconst), seasonNumber, EpisodeNumber */
SELECT TOP(3) *
FROM titleEpisode

/* title Principals -- MovieID (tconst) ordering, nconst, category (self, directory, ...), job, characters */
SELECT TOP(3) *--COUNT(*)
FROM titlePrincipals

SELECT TOP(3) *--COUNT(*)
FROM nameBasics

SELECT *
FROM titlePrincipals
WHERE tconst LIKE 'tt0120338'

/* title ratings -- MovieID (tconst), average rating, numVotes */
SELECT TOP(3) *
FROM titleRatings
WHERE tconst LIKE 'tt0120338'

/* what exactly is in the Principals table? */
/* categories == composer. actor, production_designer, archive_footage, 
self, actress, cinematographer, producer, director, editor, writer */
SELECT DISTINCT TOP(100) category
FROM titlePrincipals 

SELECT TOP(200) COUNT(DISTINCT nconst)
FROM titlePrincipals
--GROUP BY tconst

SELECT TOP(200) *
FROM titlePrincipals
ORDER BY tconst

SELECT TOP(200) *
FROM titlePrincipals
WHERE tconst = 'tt0008641'


SELECT TOP(100) *
FROM titleCrew

SELECT TOP(30) *
FROM nameBasics

PRINT 'How many actors/actresses are in the database?'
SELECT COUNT(DISTINCT primaryName)
FROM nameBasics
WHERE primaryProfession LIKE '%actor%' OR
	  primaryProfession LIKE '%actress%'

PRINT 'Are there any duplicate actors/actresses in the database?'
SELECT DISTINCT primaryName, count(primaryName) AS countActor
FROM nameBasics
WHERE primaryProfession LIKE '%actor%' OR
	  primaryProfession LIKE '%actress%'
GROUP BY primaryName ORDER BY countActor DESC

PRINT 'Check the John Williams primaryName for why there are some duplicates?'
SELECT DISTINCT *
FROM nameBasics
WHERE primaryName LIKE 'John Williams' AND (primaryProfession LIKE '%actor%' OR
	  primaryProfession LIKE '%actress%')

PRINT 'Duplicate actors are different people with the same name'

SELECT TOP(5) *
FROM titleRatings

PRINT 'Find the highest rated films with more than 20 votes '
SELECT TOP(10) *
FROM titleRatings
WHERE numVotes > 20
ORDER BY averageRating DESC

PRINT 'Find the highest rated films with more than 20 votes and determine which movie name they are'

SELECT TOP(100) titleBasics.tconst, titleBasics.titleType, titleBasics.primaryTitle, titleBasics.originalTitle, titleRatings.averageRating,
titleRatings.numVotes
FROM titleBasics
INNER JOIN titleRatings ON titleBasics.tconst=titleRatings.tconst
WHERE titleRatings.numVotes > 10000 AND titleBasics.titleType LIKE '%movie%'
ORDER BY titleRatings.averageRating DESC

PRINT 'Find the top actors in one of the top rated movies'

SELECT TOP(30) *
FROM titleBasics

SELECT primaryName, primaryProfession
FROM nameBasics
WHERE knownForTitles LIKE '%tt0111161%'

SELECT TOP(10) *
FROM nameBasics
WHERE primaryName LIKE '%Freeman%'

EXEC movieName @movieID = 'tt0097239'
EXEC personName @actorID = 'nm0476945'

SELECT TOP(5) *
FROM titleBasics
WHERE titleType = 'movie'


/* Are there movies in the titleEpisode table? Yes, there is one movie with ID = tt12590968 (Bring Out your Dead) */
SELECT TOP(50) tB.tconst, tB.titleType, tB.primaryTitle, tB.originalTitle
FROM titleBasics AS tB INNER JOIN titleEpisode AS tE ON tB.tconst=tE.parentTconst
WHERE tB.titleType='movie'

SELECT TOP(5) *
FROM titleEpisode
WHERE parentTconst = 'tt12590968'

/* set all '\N' values in seasonNumber and episodeNumber to 0 in titleEpisode table */
UPDATE titleEpisode
SET seasonNumber = '0'
WHERE seasonNumber = '\N'

UPDATE titleEpisode
SET episodeNumber = '0'
WHERE episodeNumber = '\N'

/* What is the longest running show on television? sort by seasonNumber and then episode number */
SELECT TOP(50) tB.primaryTitle, MAX(tE.seasonNumber) as SN, MAX(tE.episodeNumber) as EN,  cast(MAX(tE.episodeNumber) as int)*cast(MAX(tE.seasonNumber) as int) AS EpCount
FROM titleEpisode AS tE INNER JOIN titleBasics as tB ON tE.parentTconst=tB.tconst
GROUP BY tB.primaryTitle
HAVING cast(MAX(tE.episodeNumber) as int)*cast(MAX(tE.seasonNumber) as int) > 8000
ORDER BY EpCount DESC

-- Is King of Queens in the database? tconst = 'tt0165581'
SELECT TOP(1) tconst, titleType, primaryTitle
FROM titleBasics
WHERE primaryTitle LIKE '%king of queens%'

SELECT	seasonNumber, COUNT(episodeNumber) as num_episodes
FROM titleEpisode
WHERE parentTconst='tt0165581'
GROUP BY seasonNumber
ORDER BY seasonNumber Desc


/* FIND IMDB top 250 */
PRINT 'Find the highest rated films with more than 20 votes '
SELECT tb.primaryTitle, tr.averageRating, tr.numVotes, tb.titleType
FROM titleRatings as tr JOIN titleBasics as tb on tr.tconst = tb.tconst
WHERE tr.numVotes > 50000 and tb.titleType LIKE 'movie'
ORDER BY CAST(tr.averageRating AS FLOAT) DESC, CAST(tr.numVotes AS INT) DESC



/*----------------- Harder Questions ---------------------*/

-- Which two people have worked on the most projects together? --

-- took 25 minutes to execute All rows from CTE0 (~7million total) yields 21,565,150 Rows
With CTE0 as (
	-- Takes 1.5 minutes to execute and reduces titlePrincipals to 7 million rows
	SELECT tp.tconst, tp.nconst 
	FROM titlePrincipals as tp JOIN titleBasics as tb on tb.tconst = tp.tconst
	WHERE tb.titleType LIKE 'movie' OR tb.titleType LIKE 'tvMovie' OR tb.titleType LIKE 'video'
),
	CTE1 as (
		SELECT tp.tconst as tconst, nb.primaryName as primaryName
		FROM CTE0 as tp JOIN nameBasics as nb on nb.nconst = tp.nconst
	),
	CTE2 AS (
		SELECT c1.primaryName as name1, c2.primaryName as name2
		FROM CTE1 as c1 JOIN CTE1 as c2 on c1.tconst=c2.tconst
		WHERE c1.primaryName != c2.primaryName AND c1.primaryName < c2.primaryName 
	)
	SELECT name1, name2, COUNT(name1) as num_appearances
	FROM CTE2
	GROUP BY name1, name2
--	ORDER BY num_appearances DESC
	





	--SELECT tconst, STRING_AGG(nconst,', ') as members
	--FROM CTE1
	--GROUP BY tconst


-- took 6 minutes to execute this query
With CTE0 as (
	SELECT *
	FROM titlePrincipals
)
SELECT tp.tconst, nb.primaryName
FROM CTE0 as tp JOIN nameBasics as nb on nb.nconst = tp.nconst


-- Takes 1.5 minutes to execute and reduces titlePrincipals to 7 million rows
With CTE0 as (
	SELECT *
	FROM titlePrincipals
)
SELECT tp.tconst, tb.titleType
FROM CTE0 as tp JOIN titleBasics as tb on tb.tconst = tp.tconst
WHERE (tb.titleType LIKE 'movie' OR tb.titleType LIKE 'tvMovie' OR tb.titleType LIKE 'video') AND CAST(tb.startYear as INT) > 1920

SELECT TOP(100) *
FROM titleBasics


SELECT TOP(10) tconst, nconst
FROM titlePrincipals;


-- The following two queries are identical

-- Query One is a self join on tconst
-- Execution time = 17seconds on 1million rows of data
WITH CTE1 as (
	SELECT TOP(1000000) *
	FROM titlePrincipals
)
SELECT c1.tconst as tconst1, c2.tconst as tconst2, c1.nconst as name1, c2.nconst as name2
FROM CTE1 as c1 JOIN CTE1 as c2 on c1.tconst=c2.tconst
WHERE c1.nconst != c2.nconst AND c1.nconst < c2.nconst; 

-- Query Two is a CROSS JOIN on itself (requires filtering out any rows where c2.tconst <> c1.tconst
-- Execution time = 12seconds on 1million rows of data
WITH CTE1 as (
	SELECT TOP(1000000) *
	FROM titlePrincipals
)
SELECT c1.tconst as tconst1, c2.tconst as tconst2, c1.nconst as name1, c2.nconst as name2
FROM CTE1 as c1 CROSS JOIN CTE1 as c2
WHERE c1.nconst != c2.nconst AND c1.nconst < c2.nconst  AND c2.tconst=c1.tconst

SELECT TOP(100) *
FROM titlePrincipals;

WITH CTE1 as (
	SELECT TOP(100) *
	FROM titlePrincipals
)
SELECT *
FROM CTE1
ORDER BY tconst
FETCH NEXT 1 ROWS ONLY

