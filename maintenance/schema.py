from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from maintenance.config import FEATURES


class MachineInput(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    machine_type: Literal['L', 'M', 'H'] = 'M'
    air_temperature: float = Field(default=300.5, ge=250, le=400)
    process_temperature: float = Field(default=310.2, ge=250, le=500)
    rotational_speed: int = Field(default=1500, ge=1, le=10000)
    torque: float = Field(default=42, ge=0, le=200)
    tool_wear: int = Field(default=120, ge=0, le=1000)

    def feature_row(self):
        return dict(zip(FEATURES, [self.machine_type, self.air_temperature,
                    self.process_temperature, self.rotational_speed, self.torque, self.tool_wear]))


class BatchInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    records: list[MachineInput] = Field(min_length=1, max_length=1000)
