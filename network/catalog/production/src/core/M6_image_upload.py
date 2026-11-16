"""Build safe Earth Engine image upload commands that replace assets in place."""


def build_image_upload_command(project, asset_id, gcs_uri, metadata):
    """Build an EE CLI image upload command using --force, without asset deletion."""
    command = [
        'earthengine',
        f'--project={project}',
        'upload',
        'image',
        '--force',
        f'--asset_id={asset_id}',
        '--pyramiding_policy=mode',
        '--time_start',
        str(metadata['time_start']),
        '--time_end',
        str(metadata['time_end']),
    ]
    properties = {
        'source': metadata.get('source', 'mapbiomas-fire'),
        **{
            key: metadata[key]
            for key in (
                'model_id', 'region', 'period', 'year', 'month', 'campaign',
                'country', 'version', 'type', 'periodicity',
            )
            if key in metadata
        },
    }
    for key, value in properties.items():
        command.extend(['--property', f'{key}={value}'])
    command.append(gcs_uri)
    return command
