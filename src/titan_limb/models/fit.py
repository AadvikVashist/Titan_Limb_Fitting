"""Typed rows for saved limb-fit results."""

from pydantic import BaseModel, ConfigDict, model_validator

from titan_limb.models.core import (
    Channel,
    FitFailureReason,
    FitStatus,
    Hemisphere,
    SmoothingMethod,
)

EXPECTED_COVARIANCE_VALUES = 9


class LegacyFitRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cube_id: str
    band: int
    wavelength_um: float
    channel: Channel
    hemisphere: Hemisphere
    slant_angle_degrees: int
    profile_points: int
    status: FitStatus
    failure_reason: FitFailureReason | None = None
    smoothing_method: SmoothingMethod | None = None
    intensity_center: float | None = None
    u1: float | None = None
    u2: float | None = None
    u_sum: float | None = None
    r_squared: float | None = None
    covariance: tuple[float, ...] | None = None

    @model_validator(mode="after")
    def validate_result_state(self) -> "LegacyFitRecord":
        if self.status is FitStatus.FAILED:
            if self.failure_reason is None:
                raise ValueError("failed fit requires a failure reason")
            return self
        required_values = (
            self.smoothing_method,
            self.intensity_center,
            self.u1,
            self.u2,
            self.u_sum,
            self.r_squared,
            self.covariance,
        )
        if any(value is None for value in required_values):
            raise ValueError("successful fit requires parameters and fit statistics")
        if len(self.covariance or ()) != EXPECTED_COVARIANCE_VALUES:
            raise ValueError("quadratic covariance must contain nine values")
        return self
