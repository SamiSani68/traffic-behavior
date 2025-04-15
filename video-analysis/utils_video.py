def estimate_distance(reference_pixels, reference_meters):
    """
    Estimate pixel-per-meter ratio using a known reference.

    Args:
        reference_pixels (int): Number of pixels covering a known distance.
        reference_meters (float): Known real-world distance in meters.

    Returns:
        float: Pixel-per-meter ratio.
    """
    return reference_pixels / reference_meters
