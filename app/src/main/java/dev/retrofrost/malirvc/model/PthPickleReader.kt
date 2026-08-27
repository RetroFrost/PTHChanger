package dev.retrofrost.malirvc.model

import java.io.ByteArrayInputStream
import java.io.EOFException
import java.nio.charset.StandardCharsets
import java.util.LinkedHashMap

/**
 * Small, deliberately restricted pickle VM for PyTorch state-dict archives.
 *
 * It supports the protocol-2 opcodes emitted by RVC inference .pth files and
 * reconstructs tensors as lightweight references into the ZIP's data/ storage entries. It never executes Python code and only recognises a small allowlist
 * of GLOBAL/REDUCE targets.
 */
internal class PthPickleReader(bytes: ByteArray) {
    private val input = ByteArrayInputStream(bytes)
    private val stack = ArrayList<Any?>()
    private val memo = HashMap<Int, Any?>()
    private object Mark

    data class GlobalRef(val module: String, val name: String)

    fun read(): Any? {
        while (true) {
            when (val op = u8()) {
                0x80 -> u8() // PROTO
                0x2e -> return pop() // STOP
                0x28 -> stack.add(Mark) // MARK
                0x7d -> stack.add(LinkedHashMap<Any?, Any?>()) // EMPTY_DICT
                0x5d -> stack.add(ArrayList<Any?>()) // EMPTY_LIST
                0x29 -> stack.add(emptyList<Any?>()) // EMPTY_TUPLE
                0x4e -> stack.add(null) // NONE
                0x88 -> stack.add(true) // NEWTRUE
                0x89 -> stack.add(false) // NEWFALSE

                0x4b -> stack.add(u8()) // BININT1
                0x4d -> stack.add(u16le()) // BININT2
                0x4a -> stack.add(i32le()) // BININT

                0x58 -> { // BINUNICODE
                    val len = i32le()
                    require(len >= 0) { "Negative BINUNICODE length" }
                    stack.add(String(readN(len), StandardCharsets.UTF_8))
                }
                0x54 -> { // BINSTRING (legacy bytes/string)
                    val len = i32le()
                    stack.add(String(readN(len), StandardCharsets.ISO_8859_1))
                }
                0x55 -> { // SHORT_BINSTRING
                    stack.add(String(readN(u8()), StandardCharsets.ISO_8859_1))
                }

                0x63 -> { // GLOBAL
                    val module = readLineAscii()
                    val name = readLineAscii()
                    require(
                        (module == "collections" && name == "OrderedDict") ||
                            (module == "torch._utils" && name == "_rebuild_tensor_v2") ||
                            (module == "torch" && name.endsWith("Storage"))
                    ) { "Unsupported pickle GLOBAL $module.$name" }
                    stack.add(GlobalRef(module, name))
                }

                0x71 -> memo[u8()] = peek() // BINPUT
                0x72 -> memo[i32le()] = peek() // LONG_BINPUT
                0x68 -> stack.add(memo[u8()] ?: error("Missing BINGET memo"))
                0x6a -> stack.add(memo[i32le()] ?: error("Missing LONG_BINGET memo"))

                0x74 -> stack.add(popMarked()) // TUPLE
                0x85 -> stack.add(listOf(pop())) // TUPLE1
                0x86 -> {
                    val b = pop(); val a = pop(); stack.add(listOf(a, b))
                }
                0x87 -> {
                    val c = pop(); val b = pop(); val a = pop(); stack.add(listOf(a, b, c))
                }

                0x51 -> { // BINPERSID
                    val pid = pop() as? List<*> ?: error("Invalid persistent id")
                    require(pid.size >= 5 && pid[0] == "storage") { "Unsupported persistent id: $pid" }
                    val global = pid[1] as? GlobalRef ?: error("Invalid storage type")
                    val dtype = when (global.name) {
                        "HalfStorage" -> TorchDType.FLOAT16
                        "FloatStorage" -> TorchDType.FLOAT32
                        "LongStorage" -> TorchDType.INT64
                        "IntStorage" -> TorchDType.INT32
                        else -> TorchDType.UNKNOWN
                    }
                    val key = pid[2].toString()
                    val count = (pid[4] as Number).toLong()
                    stack.add(StorageRef(key, dtype, count))
                }

                0x52 -> { // REDUCE
                    val args = pop() as? List<*> ?: error("REDUCE args are not tuple")
                    val callable = pop() as? GlobalRef ?: error("REDUCE target is not global")
                    stack.add(reduce(callable, args))
                }

                0x75 -> { // SETITEMS
                    val items = popMarked()
                    @Suppress("UNCHECKED_CAST")
                    val map = peek() as? MutableMap<Any?, Any?> ?: error("SETITEMS target is not map")
                    require(items.size % 2 == 0)
                    var i = 0
                    while (i < items.size) {
                        map[items[i]] = items[i + 1]
                        i += 2
                    }
                }

                0x65 -> { // APPENDS
                    val items = popMarked()
                    @Suppress("UNCHECKED_CAST")
                    val list = peek() as? MutableList<Any?> ?: error("APPENDS target is not list")
                    list.addAll(items)
                }

                else -> error("Unsupported pickle opcode 0x${op.toString(16)}")
            }
        }
    }

    private fun reduce(target: GlobalRef, args: List<*>): Any? = when {
        target.module == "collections" && target.name == "OrderedDict" -> {
            val out = LinkedHashMap<Any?, Any?>()
            if (args.isNotEmpty()) {
                val pairs = args[0] as? List<*> ?: emptyList<Any?>()
                for (p in pairs) {
                    val pair = p as? List<*> ?: continue
                    if (pair.size == 2) out[pair[0]] = pair[1]
                }
            }
            out
        }
        target.module == "torch._utils" && target.name == "_rebuild_tensor_v2" -> {
            require(args.size >= 4) { "Bad _rebuild_tensor_v2 args" }
            val storage = args[0] as StorageRef
            val offset = (args[1] as Number).toLong()
            val shape = numberList(args[2])
            val stride = numberList(args[3])
            TorchTensorRef(storage, offset, shape, stride)
        }
        else -> error("Blocked REDUCE target ${target.module}.${target.name}")
    }

    private fun numberList(v: Any?): LongArray = (v as? List<*>)
        ?.map { (it as Number).toLong() }
        ?.toLongArray() ?: error("Expected integer tuple")

    private fun popMarked(): List<Any?> {
        var i = stack.lastIndex
        while (i >= 0 && stack[i] !== Mark) i--
        require(i >= 0) { "MARK not found" }
        val out = stack.subList(i + 1, stack.size).toList()
        while (stack.size > i) stack.removeAt(stack.lastIndex)
        return out
    }

    private fun pop(): Any? = stack.removeAt(stack.lastIndex)
    private fun peek(): Any? = stack.last()

    private fun u8(): Int {
        val v = input.read()
        if (v < 0) throw EOFException()
        return v
    }
    private fun u16le(): Int = u8() or (u8() shl 8)
    private fun i32le(): Int = u8() or (u8() shl 8) or (u8() shl 16) or (u8() shl 24)
    private fun readN(n: Int): ByteArray {
        val b = ByteArray(n)
        var off = 0
        while (off < n) {
            val r = input.read(b, off, n - off)
            if (r < 0) throw EOFException()
            off += r
        }
        return b
    }
    private fun readLineAscii(): String {
        val out = ArrayList<Byte>()
        while (true) {
            val b = u8()
            if (b == 0x0a) break
            out += b.toByte()
        }
        return String(out.toByteArray(), StandardCharsets.US_ASCII)
    }
}
