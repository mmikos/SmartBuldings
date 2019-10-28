SELECT
      convert(varchar, timeid, 23) as 'Date'
      ,convert(varchar, timeid, 8) as 'Hour'
    --   ,h.[spaceid]
    --   ,h.[datatypeid]
    --   ,[minval]
    --   ,[maxval]
    --   ,[countval]
    ,CASE WHEN avgval IS NULL THEN sumval
    ELSE avgval END AS 'Value'
    --   ,[sumval]  
    --   ,[avgval]
    --   ,[stdevval]
      
      ,[DataTypeName] as 'PortName'

    --   ,bu.[SpaceId]
    --   ,bu.[ParentSpaceId]
    --   ,[TimeZone]
      ,[BuildingName]
    --   ,[LEVEL]

      ,sp.[SpaceSubType]
      ,sp.[SpaceType]
      ,sp.[SpaceSurfaceAreaM2]
      ,sp.[SpaceFriendlyName]
  FROM [dbo].[fact_houraggnumeric] h
  left join [dbo].[dim_datatype] da on da.DataTypeId = h.datatypeid
  left join [dbo].[dim_space] sp on h.spaceid = sp.SpaceId
  left join [dbo].[vw_buildingspace] bu on sp.SpaceId = bu.SpaceId

WHERE convert(varchar, timeid, 23) like '2019-09%' and BuildingName like 'EDGE%'