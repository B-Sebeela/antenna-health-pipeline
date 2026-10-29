# ============================================================
# FEKO FEED OFFSET DATASET GENERATOR
# 10-SCENARIO STORAGE-CONSCIOUS VERSION
# ============================================================

$ErrorActionPreference = "Continue"

Write-Host ""
Write-Host "============================================================"
Write-Host " FEKO FEED OFFSET DATASET GENERATOR"
Write-Host "============================================================"
Write-Host ""

# ============================================================
# PATHS
# ============================================================

$ProjectRoot = $PSScriptRoot

$FekoBin = "C:\Program Files\Altair\2022\feko\bin"

$CadfekoBatch = Join-Path $FekoBin "cadfeko_batch.exe"
$Runfeko      = Join-Path $FekoBin "runfeko.exe"

$Template = "C:\Users\26250276\dipole_element_feed_offset_10percent.cfx"

$SimulationRoot = Join-Path $ProjectRoot "data\feed_offset_simulations"
$RawRoot        = Join-Path $ProjectRoot "data\raw"
$ManifestPath   = Join-Path $ProjectRoot "data\feed_offset_manifest.csv"

Write-Host "Project root:"
Write-Host "  $ProjectRoot"
Write-Host ""

Write-Host "FEKO bin:"
Write-Host "  $FekoBin"
Write-Host ""

Write-Host "Template:"
Write-Host "  $Template"
Write-Host ""

Write-Host "Simulation output:"
Write-Host "  $SimulationRoot"
Write-Host ""

Write-Host "Raw FFE output:"
Write-Host "  $RawRoot"
Write-Host ""


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if (-not (Test-Path $CadfekoBatch)) {
    Write-Host ""
    Write-Host "ERROR: CADFEKO_BATCH not found:"
    Write-Host "  $CadfekoBatch"
    exit 1
}

if (-not (Test-Path $Runfeko)) {
    Write-Host ""
    Write-Host "ERROR: RUNFEKO not found:"
    Write-Host "  $Runfeko"
    exit 1
}

if (-not (Test-Path $Template)) {
    Write-Host ""
    Write-Host "ERROR: FEKO template not found:"
    Write-Host "  $Template"
    exit 1
}


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

New-Item -ItemType Directory -Force -Path $SimulationRoot | Out-Null
New-Item -ItemType Directory -Force -Path $RawRoot | Out-Null


# ============================================================
# DATASET PLAN
#
# Offset percentage:
#
#   0%  = nominal feed at 50%
#
# Positive offset:
#   moves feed below the 50% reference
#
# Negative offset:
#   moves feed above the 50% reference
#
# Total = 10 scenarios
# ============================================================

$Scenarios = @(
    [PSCustomObject]@{
        ScenarioId = "H01_DIPOLE_FEED_50"
        Offset = 0
        FeedPosition = 50
        FaultType = "NONE"
        Direction = "NOMINAL"
    },

    [PSCustomObject]@{
        ScenarioId = "FO_DIPOLE_FEED_45"
        Offset = 5
        FeedPosition = 45
        FaultType = "FEED_OFFSET"
        Direction = "LOWER_THAN_REFERENCE"
    },

    [PSCustomObject]@{
        ScenarioId = "FO_DIPOLE_FEED_55"
        Offset = -5
        FeedPosition = 55
        FaultType = "FEED_OFFSET"
        Direction = "HIGHER_THAN_REFERENCE"
    },

    [PSCustomObject]@{
        ScenarioId = "FO_DIPOLE_FEED_40"
        Offset = 10
        FeedPosition = 40
        FaultType = "FEED_OFFSET"
        Direction = "LOWER_THAN_REFERENCE"
    },

    [PSCustomObject]@{
        ScenarioId = "FO_DIPOLE_FEED_60"
        Offset = -10
        FeedPosition = 60
        FaultType = "FEED_OFFSET"
        Direction = "HIGHER_THAN_REFERENCE"
    },

    [PSCustomObject]@{
        ScenarioId = "FO_DIPOLE_FEED_30"
        Offset = 20
        FeedPosition = 30
        FaultType = "FEED_OFFSET"
        Direction = "LOWER_THAN_REFERENCE"
    },

    [PSCustomObject]@{
        ScenarioId = "FO_DIPOLE_FEED_70"
        Offset = -20
        FeedPosition = 70
        FaultType = "FEED_OFFSET"
        Direction = "HIGHER_THAN_REFERENCE"
    },

    [PSCustomObject]@{
        ScenarioId = "FO_DIPOLE_FEED_20"
        Offset = 30
        FeedPosition = 20
        FaultType = "FEED_OFFSET"
        Direction = "LOWER_THAN_REFERENCE"
    },

    [PSCustomObject]@{
        ScenarioId = "FO_DIPOLE_FEED_80"
        Offset = -30
        FeedPosition = 80
        FaultType = "FEED_OFFSET"
        Direction = "HIGHER_THAN_REFERENCE"
    },

    [PSCustomObject]@{
        ScenarioId = "FO_DIPOLE_FEED_10"
        Offset = 40
        FeedPosition = 10
        FaultType = "FEED_OFFSET"
        Direction = "LOWER_THAN_REFERENCE"
    }
)


# ============================================================
# DISPLAY DATASET PLAN
# ============================================================

Write-Host ""
Write-Host "============================================================"
Write-Host " DATASET PLAN"
Write-Host "============================================================"
Write-Host ""

Write-Host "Number of scenarios: $($Scenarios.Count)"
Write-Host ""

Write-Host "Offset -> Feed Position"
Write-Host "-----------------------"

foreach ($Scenario in $Scenarios) {

    "{0,6}% -> {1,6}%" -f `
        $Scenario.Offset, `
        $Scenario.FeedPosition |
        Write-Host
}


# ============================================================
# RESULT STORAGE
# ============================================================

$Results = @()

$SuccessfulCount = 0
$AlreadyExistsCount = 0
$FailedCount = 0


# ============================================================
# PROCESS EACH SCENARIO
# ============================================================

for ($i = 0; $i -lt $Scenarios.Count; $i++) {

    $Scenario = $Scenarios[$i]

    $ScenarioNumber = $i + 1

    $ScenarioId = $Scenario.ScenarioId

    $Offset = $Scenario.Offset

    $FeedPosition = $Scenario.FeedPosition

    $ScenarioDir = Join-Path $SimulationRoot $ScenarioId

    $ScenarioCfx = Join-Path `
        $ScenarioDir `
        "$ScenarioId.cfx"

    $ScenarioRawDir = Join-Path `
        $RawRoot `
        $ScenarioId


    Write-Host ""
    Write-Host "============================================================"
    Write-Host " SCENARIO $ScenarioNumber / $($Scenarios.Count)"
    Write-Host "============================================================"
    Write-Host ""

    Write-Host "Scenario ID:"
    Write-Host "  $ScenarioId"

    Write-Host "FEKO variable:"
    Write-Host "  feed_offset_percentage = $Offset"

    Write-Host "Actual feed position:"
    Write-Host "  $FeedPosition%"

    Write-Host "Fault type:"
    Write-Host "  $($Scenario.FaultType)"

    Write-Host "Direction:"
    Write-Host "  $($Scenario.Direction)"

    Write-Host ""


    # ========================================================
    # CHECK IF RAW FFE ALREADY EXISTS
    # ========================================================

    $ExistingFfe = @()

    if (Test-Path $ScenarioRawDir) {

        $ExistingFfe = @(
            Get-ChildItem `
                -Path $ScenarioRawDir `
                -Filter "*.ffe" `
                -File `
                -ErrorAction SilentlyContinue
        )
    }

    if ($ExistingFfe.Count -gt 0) {

        Write-Host "FFE already exists:"
        Write-Host "  $($ExistingFfe[0].FullName)"

        Write-Host ""
        Write-Host "Skipping simulation."

        $Results += [PSCustomObject]@{
            scenario_id = $ScenarioId
            offset_percent = $Offset
            feed_position_percent = $FeedPosition
            direction = $Scenario.Direction
            status = "ALREADY_EXISTS"
        }

        $AlreadyExistsCount++

        continue
    }


    # ========================================================
    # CREATE SCENARIO DIRECTORY
    # ========================================================

    try {

        New-Item `
            -ItemType Directory `
            -Force `
            -Path $ScenarioDir |
            Out-Null

        New-Item `
            -ItemType Directory `
            -Force `
            -Path $ScenarioRawDir |
            Out-Null
    }
    catch {

        Write-Host ""
        Write-Host "ERROR: Failed to create scenario directories."
        Write-Host $_.Exception.Message

        $Results += [PSCustomObject]@{
            scenario_id = $ScenarioId
            offset_percent = $Offset
            feed_position_percent = $FeedPosition
            direction = $Scenario.Direction
            status = "DIRECTORY_FAILED"
        }

        $FailedCount++

        continue
    }


    # ========================================================
    # COPY TEMPLATE
    # ========================================================

    Write-Host "Copying FEKO template..."

    try {

        Copy-Item `
            -Path $Template `
            -Destination $ScenarioCfx `
            -Force `
            -ErrorAction Stop
    }
    catch {

        Write-Host ""
        Write-Host "ERROR: Failed to copy template."
        Write-Host $_.Exception.Message

        $Results += [PSCustomObject]@{
            scenario_id = $ScenarioId
            offset_percent = $Offset
            feed_position_percent = $FeedPosition
            direction = $Scenario.Direction
            status = "COPY_FAILED"
        }

        $FailedCount++

        continue
    }


    # ========================================================
    # APPLY FEKO VARIABLE
    #
    # IMPORTANT:
    #
    # "-#" and "feed_offset_percentage=value"
    #
    # MUST be separate arguments.
    # ========================================================

    Write-Host ""
    Write-Host "Applying FEKO variable..."
    Write-Host "  feed_offset_percentage = $Offset"
    Write-Host ""

    $CadfekoArguments = @(
        $ScenarioCfx
        "-#"
        "feed_offset_percentage=$Offset"
    )

    & $CadfekoBatch @CadfekoArguments

    $CadfekoExitCode = $LASTEXITCODE

    if ($CadfekoExitCode -ne 0) {

        Write-Host ""
        Write-Host "ERROR: CADFEKO failed."
        Write-Host "Exit code: $CadfekoExitCode"

        $Results += [PSCustomObject]@{
            scenario_id = $ScenarioId
            offset_percent = $Offset
            feed_position_percent = $FeedPosition
            direction = $Scenario.Direction
            status = "CADFEKO_FAILED"
        }

        $FailedCount++

        continue
    }

    Write-Host ""
    Write-Host "CADFEKO variable update successful."


    # ========================================================
    # RUN FEKO SOLVER
    # ========================================================

    Write-Host ""
    Write-Host "Running FEKO solver..."
    Write-Host ""

    Push-Location $ScenarioDir

    try {

        & $Runfeko "$ScenarioId" -np 4

        $RunfekoExitCode = $LASTEXITCODE
    }
    finally {

        Pop-Location
    }


    if ($RunfekoExitCode -ne 0) {

        Write-Host ""
        Write-Host "ERROR: FEKO solver failed."
        Write-Host "Exit code: $RunfekoExitCode"

        $Results += [PSCustomObject]@{
            scenario_id = $ScenarioId
            offset_percent = $Offset
            feed_position_percent = $FeedPosition
            direction = $Scenario.Direction
            status = "RUNFEKO_FAILED"
        }

        $FailedCount++

        continue
    }

    Write-Host ""
    Write-Host "FEKO solver completed."


    # ========================================================
    # SEARCH FOR FFE
    # ========================================================

    Write-Host ""
    Write-Host "Searching for FFE output..."
    Write-Host ""

    $FfeFiles = @(
        Get-ChildItem `
            -Path $ScenarioDir `
            -Recurse `
            -Filter "*.ffe" `
            -File `
            -ErrorAction SilentlyContinue
    )


    if ($FfeFiles.Count -eq 0) {

        Write-Host "ERROR: No FFE file was found."

        $Results += [PSCustomObject]@{
            scenario_id = $ScenarioId
            offset_percent = $Offset
            feed_position_percent = $FeedPosition
            direction = $Scenario.Direction
            status = "FFE_NOT_FOUND"
        }

        $FailedCount++

        continue
    }


    # ========================================================
    # COPY FFE TO RAW DIRECTORY
    # ========================================================

    $FfeSuccess = $false

    foreach ($FfeFile in $FfeFiles) {

        $DestinationFfe = Join-Path `
            $ScenarioRawDir `
            $FfeFile.Name

        try {

            Copy-Item `
                -Path $FfeFile.FullName `
                -Destination $DestinationFfe `
                -Force `
                -ErrorAction Stop

            Write-Host "FFE copied:"
            Write-Host "  $DestinationFfe"

            $FfeSuccess = $true

            break
        }
        catch {

            Write-Host ""
            Write-Host "ERROR: Failed to copy FFE."
            Write-Host $_.Exception.Message
        }
    }


    if (-not $FfeSuccess) {

        $Results += [PSCustomObject]@{
            scenario_id = $ScenarioId
            offset_percent = $Offset
            feed_position_percent = $FeedPosition
            direction = $Scenario.Direction
            status = "FFE_COPY_FAILED"
        }

        $FailedCount++

        continue
    }


    # ========================================================
    # SCENARIO SUCCESS
    # ========================================================

    Write-Host ""
    Write-Host "SCENARIO SUCCESSFUL"


    $Results += [PSCustomObject]@{
        scenario_id = $ScenarioId
        offset_percent = $Offset
        feed_position_percent = $FeedPosition
        direction = $Scenario.Direction
        status = "SUCCESS"
    }

    $SuccessfulCount++


    # ========================================================
    # IMPORTANT STORAGE CLEANUP
    #
    # The FFE has already been copied into data\raw.
    #
    # We no longer need the large FEKO simulation directory.
    # Delete it before moving to the next scenario.
    # ========================================================

    Write-Host ""
    Write-Host "Cleaning temporary FEKO simulation files..."

    try {

        Remove-Item `
            -Path $ScenarioDir `
            -Recurse `
            -Force `
            -ErrorAction Stop

        Write-Host "Temporary simulation files removed."
    }
    catch {

        Write-Host ""
        Write-Host "WARNING: Could not completely remove temporary files."
        Write-Host $_.Exception.Message
    }


    # ========================================================
    # SHOW REMAINING DISK SPACE
    # ========================================================

    $ProjectDrive = Split-Path `
        -Qualifier `
        $ProjectRoot

    $DriveName = $ProjectDrive.TrimEnd(":")

    $DriveInfo = Get-PSDrive `
        -Name $DriveName `
        -ErrorAction SilentlyContinue

    if ($null -ne $DriveInfo) {

        $FreeGB = [math]::Round(
            $DriveInfo.Free / 1GB,
            2
        )

        Write-Host ""
        Write-Host "Free disk space:"
        Write-Host "  $FreeGB GB"
    }
}


# ============================================================
# WRITE MANIFEST
# ============================================================

Write-Host ""
Write-Host "============================================================"
Write-Host " WRITING MANIFEST"
Write-Host "============================================================"
Write-Host ""

try {

    $Results |
        Export-Csv `
            -Path $ManifestPath `
            -NoTypeInformation `
            -Encoding UTF8

    Write-Host "Manifest saved:"
    Write-Host "  $ManifestPath"
}
catch {

    Write-Host ""
    Write-Host "ERROR: Could not write manifest."
    Write-Host $_.Exception.Message
}


# ============================================================
# FINAL SUMMARY
# ============================================================

Write-Host ""
Write-Host "============================================================"
Write-Host " FINAL DATASET SUMMARY"
Write-Host "============================================================"
Write-Host ""

Write-Host "Total scenarios planned: $($Scenarios.Count)"
Write-Host "Successfully generated:  $SuccessfulCount"
Write-Host "Already existed:         $AlreadyExistsCount"
Write-Host "Failed:                  $FailedCount"

Write-Host ""

Write-Host "Raw data:"
Write-Host "  $RawRoot"

Write-Host ""

Write-Host "FEKO simulations:"
Write-Host "  $SimulationRoot"

Write-Host ""

Write-Host "Manifest:"
Write-Host "  $ManifestPath"

Write-Host ""

$Results |
    Format-Table `
        scenario_id,
        offset_percent,
        feed_position_percent,
        direction,
        status `
        -AutoSize

Write-Host ""
Write-Host "============================================================"
Write-Host " DATASET GENERATION COMPLETE"
Write-Host "============================================================"
Write-Host ""