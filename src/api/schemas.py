from pydantic import BaseModel, Field, ConfigDict
from typing import Annotated


class PredictionRequest(BaseModel):
    """
    Input schema for accident severity prediction.
    """

    mois: Annotated[
        int,
        Field(
            ge=1,
            le=12,
            description="Month of the accident (1 = January, 12 = December).",
            examples=[5]
        )
    ]

    jour: Annotated[
        int,
        Field(
            ge=1,
            le=31,
            description="Day of the month when the accident occurred.",
            examples=[12]
        )
    ]

    hour: Annotated[
        int,
        Field(
            ge=0,
            le=23,
            description="Hour of the accident (0–23).",
            examples=[14]
        )
    ]

    lum: Annotated[
        int,
        Field(
            ge=0,
            description="Light conditions at the time of the accident.",
            examples=[1]
        )
    ]

    # Avoid clash with Python built-in int by using an alias
    intersection_type: Annotated[
        int,
        Field(
            alias="int",
            ge=0,
            description="Intersection-related category or context.",
            examples=[1]
        )
    ]

    atm: Annotated[
        int,
        Field(
            ge=0,
            description="Atmospheric conditions during the accident.",
            examples=[1]
        )
    ]

    col: Annotated[
        int,
        Field(
            ge=0,
            description="Collision type.",
            examples=[3]
        )
    ]

    catr: Annotated[
        int,
        Field(
            ge=0,
            description="Road category.",
            examples=[4]
        )
    ]

    circ: Annotated[
        int,
        Field(
            ge=0,
            description="Traffic circulation regime.",
            examples=[2]
        )
    ]

    nbv: Annotated[
        int,
        Field(
            ge=0,
            description="Number of lanes.",
            examples=[2]
        )
    ]

    vosp: Annotated[
        int,
        Field(
            ge=0,
            description="Special lane usage indicator.",
            examples=[0]
        )
    ]

    surf: Annotated[
        int,
        Field(
            ge=0,
            description="Road surface condition.",
            examples=[1]
        )
    ]

    infra: Annotated[
        int,
        Field(
            ge=0,
            description="Road infrastructure type.",
            examples=[0]
        )
    ]

    situ: Annotated[
        int,
        Field(
            ge=0,
            description="Situation of the accident.",
            examples=[1]
        )
    ]

    lat: Annotated[
        float,
        Field(
            ge=-90,
            le=90,
            description="Latitude of the accident location.",
            examples=[48.8566]
        )
    ]

    long: Annotated[
        float,
        Field(
            ge=-180,
            le=180,
            description="Longitude of the accident location.",
            examples=[2.3522]
        )
    ]

    place: Annotated[
        int,
        Field(
            ge=0,
            description="Seat/place of the user in the vehicle.",
            examples=[1]
        )
    ]

    catu: Annotated[
        int,
        Field(
            ge=0,
            description="User category.",
            examples=[1]
        )
    ]

    sexe: Annotated[
        int,
        Field(
            ge=0,
            description="Sex of the user.",
            examples=[1]
        )
    ]

    locp: Annotated[
        int,
        Field(
            ge=0,
            description="Pedestrian location category.",
            examples=[0]
        )
    ]

    actp: Annotated[
        int,
        Field(
            ge=0,
            description="Pedestrian action category.",
            examples=[0]
        )
    ]

    etatp: Annotated[
        int,
        Field(
            ge=0,
            description="Pedestrian physical state.",
            examples=[1]
        )
    ]

    catv: Annotated[
        int,
        Field(
            ge=0,
            description="Vehicle category.",
            examples=[7]
        )
    ]

    victim_age: Annotated[
        int,
        Field(
            ge=0,
            le=120,
            description="Estimated age of the victim.",
            examples=[35]
        )
    ]

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "mois": 5,
                "jour": 12,
                "hour": 14,
                "lum": 1,
                "int": 1,
                "atm": 1,
                "col": 3,
                "catr": 4,
                "circ": 2,
                "nbv": 2,
                "vosp": 0,
                "surf": 1,
                "infra": 0,
                "situ": 1,
                "lat": 48.8566,
                "long": 2.3522,
                "place": 1,
                "catu": 1,
                "sexe": 1,
                "locp": 0,
                "actp": 0,
                "etatp": 1,
                "catv": 7,
                "victim_age": 35
            }
        }
    )
    
class PredictionResponse(BaseModel):
    """
    Output schema for accident severity prediction.
    """

    prediction: int = Field(
        ...,
        description="Predicted severity class as numeric code.",
        examples=[2]
    )

    severity: str = Field(
        ...,
        description="Human-readable severity label.",
        examples=["Serious injury"]
    )

    description: str = Field(
        ...,
        description="Explanation of the predicted severity class.",
        examples=["Predicted as an accident with serious injuries."]
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score of the predicted class.",
        examples=[0.82]
    )

    probabilities: dict[str, float] = Field(
        ...,
        description="Probability distribution over all severity classes.",
        examples=[{
            "no_injury_minor": 0.05,
            "slight_injury": 0.10,
            "serious_injury": 0.82,
            "fatal": 0.03
        }]
    )