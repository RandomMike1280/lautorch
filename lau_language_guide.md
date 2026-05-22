# LAU Language Guide for Drone Farming

This guide covers the basics of the `.lau` scripting language used for automating drone farming based on the available examples.

## Basic Syntax

### Variables
Declare variables using the `varol` keyword:
```lau
varol item = player.getItem(1)
varol x, z = drone.getPosition() -- Multiple assignment is supported
```

### Comments
Use `--` for single-line comments. They must be on a separate line or at the end of a statement.
```lau
-- This is a comment
varol result = math.abs(-50) -- Prints 50
```

### Strings
Strings are joined (concatenated) using the `+` operator, not the `..` operator typically found in Lua.
```lau
print("Date: " + task.date())
```

### Lists and Dictionaries
`.lau` uses a unified structure for both ordered lists (arrays) and key-value pairs (dictionaries). Note that lists are **1-indexed** (the first element is at index 1).

**Defining Lists:**
```lau
-- Dictionary style
varol droneData = {
    ["Speed"] = 15,
    ["Mode"] = "Automatic"
}

-- Array style (Ordered List)
varol fruits = {"Apple", "Pear", "Banana"}
```

**Accessing and Modifying:**
You do not use `varol` when updating an existing list.
```lau
varol inventory = {"Wheat", "Corn", "Tomato"}
print(inventory[2]) -- Retrieves "Corn"
print(#inventory) -- The '#' operator gets the length of the list

inventory.new = "Potato" -- Add/update using dot notation
inventory[1 + 2] = "Watermelon" -- Add/update dynamically using index (updates 3rd item)

-- Deleting elements uses 'null' instead of 'nil'
droneData.Speed = null
```

### Control Flow
The language primarily uses Lua-like block structures (`if / then / end`, `while true do`). While C-like syntax elements (`if (condition) { ... }`) are supported, they are known to be poorly implemented and buggy. **Always prefer the Lua style for stability.**

**Logical and Relational Operators:**
*   Use uppercase `AND`, `OR`, and `NOT` for logical conditions.
*   Use `~=` for "not equal" (e.g., `if count ~= 5 then`).

```lau
if item AND it.Type == "Seed" then
    -- code
elseif NOT item then
    -- code
end

while true do
    -- code
end

-- Numeric For Loop
for i = 1, 5 do 
    print("Number:", i) 
end 

-- For Loop over Lists (use 'inpairs')
varol fruits = {"Apple", "Pear", "Cherry"} 
for index, fruit inpairs(fruits) do 
    print(index + ". fruit: " + fruit) 
end
```

### Defining Functions
You can define standard functions or assign anonymous functions to variables using the `func` keyword and closing with `end`.

```lau
-- Standard definition
func add(a, b) 
    return a + b 
end 

-- Anonymous function assigned to a variable
varol multiply = func(x, y) 
    return x * y 
end 
```

### Functions and Calling
To call a function, you must use parentheses. If you omit parentheses, you are referencing the function, not calling it.
```lau
drone.move(Enum.Direction.East) -- CORRECT: Calls the function
varol myMove = drone.move -- Assigns the function reference to a variable
myMove(Enum.Direction.South) -- Calls the referenced function
```

### Modules and Imports
You can split your code into separate `.laum` module scripts and import them into your main script using the `req()` function.
```lau
-- Import a module
varol farmingModule = req("FarmingOperations.laum")
```

## Pragmas and Asynchronous Execution

Pragmas are special instructional commands that tell the `.lau` engine how to process your code. They must be placed at the **very top** of your main `.lau` script (they do not work inside `.laum` module scripts).

### The `--!ndrone` Pragma
By default, `.lau` operates synchronously. When you issue a drone command (like `drone.doFlip()`), the script pauses until the animation finishes. 

Adding `--!ndrone` at the top of your script enables **Asynchronous (Non-Blocking) mode**. The engine will issue the command and instantly skip to the next line.

```lau
--!ndrone
drone.doFlip()
print("hi") -- Prints instantly while the drone is still flipping!
```

### The Overlapping Problem
In async mode, if you issue a command while the drone is busy, **the new command is completely ignored.**

```lau
--!ndrone
drone.doFlip() -- Starts flipping
drone.doFlip() -- IGNORED! Drone is already busy.
```

To safely use async mode, you must manually check the drone's status using `drone.status()`:
```lau
--!ndrone

while true do
    -- Only send commands if the drone is resting
    if drone.status() == Enum.DroneStatus.Sleep then
        drone.doFlip()
    end
    
    -- You can run other background calculations here!
    
    task.wait(0.1) -- Always include a small wait to prevent crashes in tight loops
end
```

## Core Objects and APIs

### The `drone` Object
Controls the automation drone's actions and retrieves data about its current tile.

*   **Farming Actions**
    *   `drone.plant(Enum.Seed.[Type])`: Plants a specific seed on the current tile.
    *   `drone.canCrop()`: Returns a boolean indicating if the plant on the current tile can be cropped (cut from root).
    *   `drone.crop()`: Collects crops like Pumpkin, Wheat, Potato, etc.
    *   `drone.canHarvest()`: Returns a boolean indicating if a fruit-bearing tree can be harvested.
    *   `drone.harvest()`: Collects fruit from fruit-bearing trees.
*   **Plant Data**
    *   `drone.getPlant()`: Returns a plant object containing properties like `HasFruit`.
    *   `drone.getPlantHasFruit()`: Returns a boolean indicating if the plant has fruit.
    *   `drone.getPlantPercent()`: Returns the growth percentage of the plant itself.
    *   `drone.getFruitPercent()`: Returns the growth percentage of the fruit.
*   **Movement & Position**
    *   `drone.move(Enum.Direction.[Direction])`: Moves the drone one unit in the specified direction (only North, South, East, West).
    *   `drone.doFlip()`: Makes the drone perform a backflip.
    *   `drone.getPosition()`: Returns both X and Z coordinates (`varol x, z = drone.getPosition()`).
    *   `drone.getPositionX()`: Returns only the X coordinate.
*   **State & Status**
    *   `drone.status()`: Returns the drone's current state (e.g., `Enum.DroneStatus.Busy` or `Enum.DroneStatus.Sleep`).
    *   `drone.useItem(Enum.Gear.[GearType])`: Commands the drone to use an item, such as a Watering Can.

### The `droneV2` Object
The V2 drone has advanced movement and tile inspection capabilities that read machine and soil buff data.
*   **Advanced Movement**
    *   `droneV2.goto(x, z)`: Commands the drone to travel directly to the specified X and Z coordinates.
    *   `droneV2.swap(Enum.Direction)`: Swaps positions with the plant (or empty space) on the adjacent tile.
*   **Tile Inspection & Gear Data**
    *   `droneV2.isLocked()`: Returns a boolean indicating if the plant on the tile is locked (cannot be swapped).
    *   `droneV2.hasGear()`: Returns a boolean indicating if a machine is currently placed on the tile.
    *   `droneV2.getGear()`: Returns a comprehensive object containing all machine details and soil buff data.
    *   `droneV2.getGearName()`: Returns the specific name of the gear.
    *   `droneV2.getGearDuration()`: Returns its remaining active duration in seconds.
*   **Soil Buffs**
    *   `droneV2.getFertilizer()` / `droneV2.getManualWater()` / `droneV2.getMachineWater()`: Returns an object with `Duration` (remaining seconds) and `Multi` (effectiveness multiplier).
    *   `droneV2.getLightning()`: Returns the remaining duration of the lightning rod protection effect in seconds.

### The `player` Object
Accesses player inventory, stats, and UI interactions.

*   **Inventory & Wealth**
    *   `player.getItem(slotNumber)`: Returns the item in the specified inventory slot (e.g., slot 1 is the first hotbar slot). The item object has properties like `Type`, `Name`, and `Amount`.
    *   `player.scrap()`: Returns the total amount of scrap (currency) the player owns.
    *   `player.calculateFinalScrap(basePrice)`: Returns the actual scrap earned after multipliers (e.g., events).
    *   `player.getTileNumber()`: Returns the player's land size (upgrade level).
*   **UI & Events**
    *   `player.alert("Message")`: Displays an alert message to the player.
    *   `player.chatted:connect(func(message) ... end)`: Event listener triggered when the player types a chat command.

### The `market` Object
Handles purchasing seeds, selling items, and market events.

*   **Market Data**
    *   `market.getSeedStock()`: Returns current seed stock.
    *   `market.getSeedPrice(Enum.Seed.[Type])`: Returns the price of a specific seed.
    *   `market.getSeedStockTime()` / `market.getGearStockTime()`: Returns time remaining for seed or gear stock refresh.
    *   `market.whatValue(slotNumber)`: Returns the market value of the item in the specified inventory slot.
*   **Transactions**
    *   `market.buySeed(Enum.Seed.[Type])` / `market.buyGear(Enum.Gear.[GearType])`: Buys a specific seed or gear.
    *   `market.sellItem(slotNumber)`: Sells the item in the specified inventory slot.
    *   `market.sellAllItem()`: Sells all sellable items from the inventory at once.
*   **Events**
    *   `market.changedSeedStock:connect(func() ... end)`: Triggered when seed stock refreshes.
    *   `market.changedGearStock:connect(func() ... end)`: Triggered when gear stock refreshes.

### The `task` Object
Provides utility functions for time and yielding. Note that loops do *not* strictly require yielding to prevent crashes, but `task.wait()` is available if needed.

*   `task.wait(seconds)`: Pauses the script for the specified number of seconds.
*   `task.date()`: Returns the current date and time as a string.
*   `task.clock()`: Returns a high-precision timestamp (useful for benchmarking code performance: `varol start = task.clock()`).

## Enums

### `Enum.Seed`
Available seed types for planting and purchasing:
`Apple`, `Bamboo`, `Banana`, `Blueberry`, `Bush`, `Cacao`, `Cactus`, `Carrot`, `Coconut`, `Corn`, `Garlic`, `Glttch`, `Grape`, `Lemon`, `Lotus`, `Mango`, `Mushroom`, `Onion`, `Pear`, `Pepper`, `Pineapple`, `Potato`, `Pumpkin`, `Strawberry`, `Tomato`, `Tree`, `Watermelon`, `Wheat`

*(Note: Use the standard seed name like `Enum.Seed.Apple`. Variations like `Enum.Seed.AppleTree` are incorrect and will not work).*

### `Enum.Direction`
Used for drone movement. There are exactly 4 available directions:
`Enum.Direction.North`, `Enum.Direction.East`, `Enum.Direction.South`, `Enum.Direction.West`.

### `Enum.DroneStatus`
Used to check if the drone is ready for a command, especially in async mode.
*   `Enum.DroneStatus.Busy`: The drone is currently performing an action (like moving or flipping).
*   `Enum.DroneStatus.Sleep`: The drone is idle and ready to receive a new command.

### `Enum.Gear`
Represents tools or machines.
*   `Enum.Gear.WateringCan`: Used to manually water tiles via `drone.useItem()`.

## Built-in Functions
*   `print("Message")`: Prints text to the console.
*   `tonumber(string)`: Converts a string to a numeric value.

### The `string` Module
*   `string.find(str, substring)`: Returns the starting index of the substring within the string (1-indexed).
*   `string.sub(str, startIndex)`: Returns a substring starting from the specified index.

### The `math` Module
*   `math.random(min, max)`: Generates a random integer between the two specified numbers.
*   `math.round(number)`: Rounds a decimal number to the nearest integer (e.g., 4.6 -> 5).
*   `math.abs(number)`: Returns the absolute value (positive form) of the number. Useful for distances.
*   `math.pi`: Returns the mathematical constant Pi (3.1415...).
